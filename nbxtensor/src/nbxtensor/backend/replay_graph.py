"""Phase 4b — per-bucket CUDA-graph capture of a FrozenPlan.

Captures a plan's recorded action list ONCE into a CUDA graph on a
dedicated non-blocking stream and replays it as one cuGraphLaunch per
step. Adoption is gated on a measured positive V100 A/B delta vs the
direct C-launcher replay (Phase 4a); every CUresult is checked and any
failure refuses LOUDLY back to direct replay for that bucket
(llama.cpp guard discipline — their CC<8 disable is a policy under a
re-capture-per-token cost model, not a hardware gate; sourced research
in optimization_engine_scoping.md §4b). Kill-switch: unset
NBX_REPLAY_GRAPH (the graph path is opt-in).

Capture legality:
- kernels are re-issued through the SAME compiled C launcher but on
  the CAPTURE stream (the recorded stream may be the uncapturable
  legacy stream; serialization semantics are preserved — one stream
  either way). The launcher is capture-safe: arg pack + cuLaunchKernel,
  no sync/query; JIT warmup already happened at recording.
- recorded D2D memcpys / memsets become cuMemcpyDtoDAsync /
  cuMemsetD8Async on the capture stream.
- recorded H2D snapshot actions become DEVICE-staged D2D nodes (the
  snapshot bytes are plan constants — uploaded once outside capture).
- the per-step INPUT copies (slot scan) and OUTPUT fresh copies stay
  OUTSIDE the graph in FrozenPlan.replay — they are the step-varying
  host-driven part.

ORDERING (the 2026-08-17 root cause): the engine's per-step input
copies are ASYNC legacy-stream work, and a NON_BLOCKING stream does
not implicitly wait for the legacy stream — an unbarriered graph
launch read one-step-STALE inputs (deterministic-looking token
doubling, masked by any diagnostic D2H sync, which made every
bisection "boundary" a timing artifact until the dump-run-correct /
plain-run-wrong pair exposed the observer effect). launch() therefore
brackets cuGraphLaunch with device-wide syncs: before (inputs land)
and after (legacy consumers read the graph's writes).

During stream capture NOTHING executes — the actions are recorded into
the graph. The capturing step therefore launches the instantiated
graph immediately after capture to actually perform its work.

R33: ctypes on libcuda only.
"""

from __future__ import annotations

import ctypes
from typing import List, Tuple

from nbxtensor.memory.allocator import DeviceAllocator

_CUDA_SUCCESS = 0
_CU_STREAM_NON_BLOCKING = 1
_CU_STREAM_CAPTURE_MODE_RELAXED = 2


class GraphReplayError(RuntimeError):
    pass


def _ck(rc: int, what: str) -> None:
    if rc != _CUDA_SUCCESS:
        raise GraphReplayError(f"{what}: CUresult={rc}")


class _CudaGraphAPI:
    """Lazy ctypes surface over libcuda's graph entry points. Every
    pointer crosses as c_void_p / c_size_t (no implicit c_int
    truncation)."""

    _inst = None

    def __init__(self) -> None:
        lib = ctypes.CDLL("libcuda.so.1")

        def bind(name, argtypes):
            fn = getattr(lib, name)
            fn.argtypes = argtypes
            fn.restype = ctypes.c_int
            return fn

        vp, sz = ctypes.c_void_p, ctypes.c_size_t
        self.begin = bind("cuStreamBeginCapture_v2", [vp, ctypes.c_int])
        self.end = bind("cuStreamEndCapture", [vp, ctypes.POINTER(vp)])
        try:
            self.instantiate = bind(
                "cuGraphInstantiateWithFlags",
                [ctypes.POINTER(vp), vp, ctypes.c_ulonglong])
            self.inst_flags = True
        except AttributeError:
            self.instantiate = bind(
                "cuGraphInstantiate_v2",
                [ctypes.POINTER(vp), vp, vp, vp, sz])
            self.inst_flags = False
        self.launch = bind("cuGraphLaunch", [vp, vp])
        self.graph_destroy = bind("cuGraphDestroy", [vp])
        self.exec_destroy = bind("cuGraphExecDestroy", [vp])
        self.stream_create = bind(
            "cuStreamCreate", [ctypes.POINTER(vp), ctypes.c_uint])
        self.stream_sync = bind("cuStreamSynchronize", [vp])
        self.stream_destroy = bind("cuStreamDestroy_v2", [vp])
        self.memcpy_dtod = bind("cuMemcpyDtoDAsync_v2", [vp, vp, sz, vp])
        self.memset_d8 = bind(
            "cuMemsetD8Async", [vp, ctypes.c_ubyte, sz, vp])
        self.get_nodes = bind(
            "cuGraphGetNodes", [vp, vp, ctypes.POINTER(sz)])
        self.ctx_sync = bind("cuCtxSynchronize", [])

    @classmethod
    def get(cls) -> "_CudaGraphAPI":
        if cls._inst is None:
            cls._inst = _CudaGraphAPI()
        return cls._inst


class CapturedPlan:
    """One bucket's action list as an instantiated CUDA graph."""

    __slots__ = ("api", "stream", "graph", "gexec", "staging",
                 "launches")

    def __init__(self, records: List[Tuple[str, tuple]],
                 dev_idx) -> None:
        api = _CudaGraphAPI.get()
        self.api = api
        self.staging: List[int] = []
        self.graph = ctypes.c_void_p()
        self.gexec = ctypes.c_void_p()
        if dev_idx is not None:
            DeviceAllocator.set_device(dev_idx)
            DeviceAllocator.ensure_triton_device(dev_idx)
        stream = ctypes.c_void_p()
        _ck(api.stream_create(ctypes.byref(stream),
                              _CU_STREAM_NON_BLOCKING), "cuStreamCreate")
        self.stream = stream
        n_launch = 0
        capturing = False
        # Pre-pass OUTSIDE capture: stage every recorded H2D snapshot
        # into a dedicated device buffer (sync malloc/memcpy are not
        # capture-legal; the snapshots are plan constants, so one
        # upload serves every launch).
        staged = {}
        for i, (tag, rec) in enumerate(records):
            if tag == "h":
                dst, snap = rec
                stage = DeviceAllocator.malloc_cuda(len(snap), dev_idx)
                buf = (ctypes.c_char * len(snap)).from_buffer_copy(snap)
                DeviceAllocator.memcpy(stage, ctypes.addressof(buf),
                                       len(snap), 1)
                staged[i] = stage
                self.staging.append(int(stage))
        try:
            _ck(api.begin(stream, _CU_STREAM_CAPTURE_MODE_RELAXED),
                "cuStreamBeginCapture_v2")
            capturing = True
            for i, (tag, rec) in enumerate(records):
                if tag == "k":
                    kernel, g0, g1, g2, _rec_stream, flat = rec
                    kernel.run(g0, g1, g2, stream.value, kernel.function,
                               kernel.packed_metadata, None, None, None,
                               *flat)
                    n_launch += 1
                elif tag == "m":
                    dst, src, nbytes, _kind = rec
                    _ck(api.memcpy_dtod(
                        ctypes.c_void_p(dst), ctypes.c_void_p(src),
                        nbytes, stream), "cuMemcpyDtoDAsync")
                elif tag == "h":
                    # Recorded H2D snapshots become DEVICE-staged D2D
                    # nodes (staged in the pre-pass above): the graph
                    # copies device-to-device — cheaper per launch than
                    # host-pinned reads, and keeps the graph free of
                    # host-memory nodes.
                    dst, snap = rec
                    _ck(api.memcpy_dtod(
                        ctypes.c_void_p(dst),
                        ctypes.c_void_p(staged[i]),
                        len(snap), stream), "cuMemcpyDtoDAsync(h2d)")
                else:  # "s"
                    ptr, value, nbytes = rec
                    _ck(api.memset_d8(
                        ctypes.c_void_p(ptr), value & 0xFF, nbytes,
                        stream), "cuMemsetD8Async")
            _ck(api.end(stream, ctypes.byref(self.graph)),
                "cuStreamEndCapture")
            capturing = False
            if not self.graph.value:
                raise GraphReplayError("cuStreamEndCapture: NULL graph")
            n_nodes = ctypes.c_size_t(0)
            _ck(api.get_nodes(self.graph, None, ctypes.byref(n_nodes)),
                "cuGraphGetNodes")
            if n_nodes.value != len(records):
                raise GraphReplayError(
                    f"captured node count {n_nodes.value} != "
                    f"{len(records)} recorded actions — capture lost "
                    f"or merged nodes")
            if api.inst_flags:
                _ck(api.instantiate(ctypes.byref(self.gexec), self.graph,
                                    0), "cuGraphInstantiateWithFlags")
            else:
                _ck(api.instantiate(ctypes.byref(self.gexec), self.graph,
                                    None, None, 0), "cuGraphInstantiate_v2")
        except BaseException:
            if capturing:
                # Terminate the dangling capture; error intentionally
                # ignored — we are already refusing.
                api.end(stream, ctypes.byref(self.graph))
            self.destroy()
            raise
        self.launches = n_launch

    def launch(self) -> None:
        # DEVICE-wide sync BEFORE the graph: the per-step input copies
        # (replay's slot scan) ride the LEGACY stream as async work,
        # and a NON_BLOCKING capture stream does NOT wait for the
        # legacy stream — without this barrier the graph read STALE
        # inputs (measured on r535/V100: one-step-lagged token
        # doubling, masked by any diagnostic D2H sync — the entire
        # 2026-08-17 bisection's "clean/broken boundaries" were timing
        # artifacts of that race).
        _ck(self.api.ctx_sync(), "cuCtxSynchronize(pre)")
        _ck(self.api.launch(self.gexec, self.stream), "cuGraphLaunch")
        # ...and AFTER: downstream consumers (output copies, sampler)
        # are legacy-stream work that would not wait for S either. A
        # failure HERE means the graph's work was already submitted —
        # the device state (e.g. KV position counters) has advanced,
        # so re-running the action list would double-advance it: mark
        # the error as post-execution so the caller crashes loudly
        # instead of "recovering" into corrupt state.
        try:
            _ck(self.api.ctx_sync(), "cuCtxSynchronize(post)")
        except GraphReplayError as e:
            e.executed = True
            raise

    def destroy(self) -> None:
        api = self.api
        if self.gexec.value:
            api.exec_destroy(self.gexec)
            self.gexec = ctypes.c_void_p()
        if self.graph.value:
            api.graph_destroy(self.graph)
            self.graph = ctypes.c_void_p()
        for p in self.staging:
            if p:
                # device staging buffers (see the "h" action handler)
                DeviceAllocator.free_cuda(p)
        self.staging = []
        if self.stream.value:
            api.stream_destroy(self.stream)
            self.stream = ctypes.c_void_p()


class SegmentedPlan:
    """A multi-card plan's action list as one CUDA graph per stretch of actions on one card (S3, 2026-10-28).

    A plan recorded across cards (an expert store's card-held tail: each layer's attention on the home card, its MoE
    op on the card that holds the block, the hidden state crossing by a peer copy) switches cards with a recorded
    SETDEV and orders the crossings with recorded device syncs. Each maximal stretch of kernels, copies and memsets
    between two of those is captured on ITS card (`CapturedPlan`, bracketed by that card's syncs, so a peer copy it
    issues has landed before the next stretch starts); the SETDEVs and syncs between stretches run directly. The order
    is the recorded order and every stretch completes before the next begins — the direct replay's semantics, one
    graph launch per stretch instead of one Python-driven launch per kernel."""

    __slots__ = ("steps", "launches", "stretches")

    def __init__(self, records: List[Tuple[str, tuple]], dev_idx) -> None:
        self.steps: List[Tuple[str, object]] = []
        self.launches = 0
        cur = dev_idx
        run: List[Tuple[str, tuple]] = []

        def close():
            if run:
                plan = CapturedPlan(list(run), cur)
                self.steps.append(("g", (cur, plan)))
                self.launches += plan.launches
                run.clear()
        try:
            for tag, rec in records:
                if tag == "d":
                    if rec[0] == cur:
                        continue            # already this card: the stretch goes on (each graph sets its own card)
                    close()
                    cur = rec[0]
                    self.steps.append(("d", rec[0]))
                elif tag == "y":
                    close()
                    self.steps.append(("y", rec[0]))
                elif tag in ("k", "m", "h", "s"):
                    run.append((tag, rec))
                else:
                    raise GraphReplayError(f"action {tag!r} has no graph form")
            close()
        except BaseException:
            self.destroy()
            raise
        self.stretches = sum(1 for t, _ in self.steps if t == "g")

    def launch(self) -> None:
        for tag, val in self.steps:
            if tag == "g":
                dev, plan = val
                if dev is not None:
                    DeviceAllocator.set_device(dev)
                    DeviceAllocator.ensure_triton_device(dev)
                plan.launch()
            elif tag == "d":
                DeviceAllocator.set_device(val)
                DeviceAllocator.ensure_triton_device(val)
            else:
                cur = DeviceAllocator.get_device()
                if cur != val:
                    DeviceAllocator.set_device(val)
                DeviceAllocator.sync_device()
                if cur != val:
                    DeviceAllocator.set_device(cur)

    def destroy(self) -> None:
        for tag, val in self.steps:
            if tag == "g":
                val[1].destroy()
        self.steps = []
