"""Frozen-plan replayer for the triton hot loop (Phase 4a: E1+E2+E6).

Removes the per-launch Python band (wrapper dtype protocol, arg
resolvers, allocation, autotune lookup, triton's Python launcher —
measured ~0.57 ms/launch on the Ming denoiser) by recording the FINAL
launch tuples of one execution and replaying them as direct C-launcher
calls. Design + byte/lifecycle contract:
docs/internal/optimization_engine_scoping.md "Phase 4a".

Per-bucket state machine (bucket = shape/symbol/interceptor signature):
  1st run  MEASURE  — normal execution, autotune warms (E2), the
                      malloc/free watermark is measured (E6 sizing).
  2nd run  RECORD   — normal machinery executes under three seams:
                      every device allocation is served from ONE slab
                      (free-list reuse mirroring the run's own free
                      order — replay-safe because the launch order is
                      identical), every kernel launch's FINAL tuple is
                      captured with tensor args flattened to RAW
                      POINTER INTS (no NBXTensor retained — the
                      recorded run's intermediates die on schedule, so
                      the slab footprint equals a NORMAL run's peak,
                      not the no-kill footprint: probe 1 measured
                      31.3 GB retained vs ~13 GB normal on Sana DiT),
                      and D2D/H2D memcpys+memsets are captured as
                      actions.
  3rd+     REPLAY   — copy new inputs into the frozen input buffers,
                      iterate the action list as direct C calls,
                      restore the recorded arena view.

Plan-breakers (loud, capability-gate class — the graph keeps the
normal path forever): any host read of device data during recording
(NBXTensor.item, D2H memcpy), slab exhaustion after the data-driven
retry, any exception mid-recording (the step is then re-executed
normally — a recording failure never corrupts a user run).

R33: zero torch. The compiled engine is untouched (D2-legit; its
launch band is ~10x smaller — Phase 4b may revisit it separately).
"""

from __future__ import annotations

import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator

ENABLED = os.environ.get("NBX_TRITON_REPLAY") == "1"

# E2 (persistable part, v1): per-bucket slab sizes survive the process
# so a warm restart goes straight to the recording pass with the right
# slab (saves the measure step AND the shortfall-retry step; probe run
# burned 3 of 12 steps cold). Launch records themselves are
# process-local by nature (kernel handles + device pointers) and are
# re-recorded once per process; triton's own cache_results=True disk
# cache already persists autotune selections across processes. The
# full E2 artifact (configs keyed by OUR fingerprint, seeded into the
# Autotuner caches, immune to source-hash invalidation) is the named
# next increment.
_SIZE_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".neurobrix", "replay_cache")


def _size_cache_path(seq) -> str:
    comp = str((seq.dag or {}).get("component_name", "component"))
    return os.path.join(_SIZE_CACHE_DIR, f"{comp}.json")


def _sig_hash(sig: tuple) -> str:
    import hashlib
    return hashlib.sha256(repr(sig).encode()).hexdigest()[:16]


def _load_slab_size(seq, sig: tuple) -> Optional[int]:
    import json
    try:
        with open(_size_cache_path(seq)) as f:
            return json.load(f).get(_sig_hash(sig))
    except (OSError, ValueError):
        return None


def _store_slab_size(seq, sig: tuple, nbytes: int) -> None:
    import json
    try:
        os.makedirs(_SIZE_CACHE_DIR, exist_ok=True)
        path = _size_cache_path(seq)
        data = {}
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {}
        data[_sig_hash(sig)] = int(nbytes)
        with open(path, "w") as f:
            json.dump(data, f)
    except OSError:
        pass  # cache is an optimization, never a failure source

_KERNEL = "k"     # (kernel, g0, g1, g2, stream, flat_vals)
_MEMCPY = "m"     # (dst, src, nbytes, kind) — device-side pointers only
_MEMSET = "s"     # (ptr, value, nbytes)
_SETDEV = "d"     # (device_idx,) — the card the following actions run on.
                  # Emitted only when the device CHANGES, so a single-device
                  # plan carries one of these and costs nothing. Without it a
                  # replay runs every recorded launch on whatever card happens
                  # to be current: silently wrong, not an error. One of the two
                  # reasons multi-device was locked out of replay.
_H2D = "h"        # (dst, host_snapshot: bytes) — the recorded host
                  # source is transient (probe 5: a dangling 16-B host
                  # staging pointer at replay); the plan owns a byte
                  # snapshot instead. A step-VARYING H2D would replay
                  # stale bytes — the byte gates adjudicate that class
                  # (contract: byte mismatch = replay bug/refusal).

_ALIGN = 512      # matches typical device allocation granularity


# ---------------------------------------------------------------------------
# E6 — slab allocator (recording-time allocation backing)
# ---------------------------------------------------------------------------


class SlabAllocator:
    """First-fit free-list allocator over ONE device slab.

    The recording run's malloc/free sequence drives carving/reuse; the
    replay issues the same launches in the same order, so every reuse
    the recorded run performed is serialization-safe at replay.
    """

    def __init__(self, nbytes: int, dev_idx: Optional[int]) -> None:
        self.size = int(nbytes)
        self.dev_idx = dev_idx
        self.base = _ORIG_MALLOC(self.size, dev_idx)
        self.free_list: List[Tuple[int, int]] = [(0, self.size)]
        self.allocs: Dict[int, Tuple[int, int]] = {}  # ptr -> (off, sz)
        self.shortfall = 0  # aligned bytes we could not serve
        # A retired slab (failed recording attempt / invalidated plan)
        # must NOT cudaFree while tensors carved from it are alive:
        # their finalizers free slab-interior pointers, which the seam
        # routes back here. Actual release happens when the last
        # outstanding alloc returns. Probe 2 proved the hazard: an
        # eager release left the text-encoder outputs dangling
        # ("Pointer argument cannot be accessed" + invalid cudaFree of
        # a slab-interior pointer).
        self.retired = False
        _ACTIVE_SLABS.append(self)

    def malloc(self, nbytes: int) -> int:
        need = (int(nbytes) + _ALIGN - 1) // _ALIGN * _ALIGN
        for i, (off, sz) in enumerate(self.free_list):
            if sz >= need:
                if sz == need:
                    self.free_list.pop(i)
                else:
                    self.free_list[i] = (off + need, sz - need)
                ptr = self.base + off
                self.allocs[ptr] = (off, need)
                return ptr
        self.shortfall += need
        return 0

    def free(self, ptr: int) -> bool:
        entry = self.allocs.pop(ptr, None)
        if entry is None:
            return False
        off, sz = entry
        self.free_list.append((off, sz))
        self.free_list.sort()
        merged: List[Tuple[int, int]] = []
        for o, s in self.free_list:
            if merged and merged[-1][0] + merged[-1][1] == o:
                merged[-1] = (merged[-1][0], merged[-1][1] + s)
            else:
                merged.append((o, s))
        self.free_list = merged
        if self.retired and not self.allocs:
            self._release_now()
        return True

    def retire(self) -> None:
        """Deferred release: cudaFree only after the last outstanding
        carved allocation has been returned by its owner's finalizer."""
        self.retired = True
        if not self.allocs:
            self._release_now()

    def _release_now(self) -> None:
        if self.base:
            _ORIG_FREE(self.base)
            self.base = 0
        if self in _ACTIVE_SLABS:
            _ACTIVE_SLABS.remove(self)


class StepSlab(SlabAllocator):
    """Per-sequence slab RECYCLED at every decode step (P-REPLAY-KV-
    DECODE B1). reset() returns the whole range to the free list so
    the next step's allocations carve IDENTICAL addresses (the decode
    graph's allocation order is structurally constant — census-
    proven); allocations from previous steps become 'stale' and their
    later finalizer frees are absorbed silently (the range is already
    recycled — safe because each step's consumers run before the next
    step begins in the synchronous decode timeline)."""

    def __init__(self, nbytes: int, dev_idx: Optional[int]) -> None:
        super().__init__(nbytes, dev_idx)
        self.stale: set = set()
        self.carve_sizes: List[int] = []   # diagnosis: per-step size seq
        self.prev_carve: Optional[List[int]] = None

    def malloc(self, nbytes: int) -> int:
        """PURE BUMP carving — mid-step frees are absorbed without
        recycling (see free below), so placement is a pure function of
        the allocation-size sequence (measured identical step-to-step)
        and IMMUNE to free-timing nondeterminism (GC-cycle finalizers
        fire at variable points; first-fit reuse turned that into
        per-step placement churn — measured: identical size sequences,
        differing offsets)."""
        self.carve_sizes.append(int(nbytes))
        need = (int(nbytes) + _ALIGN - 1) // _ALIGN * _ALIGN
        off, sz = self.free_list[0]
        if sz < need:
            self.shortfall += need
            return 0
        self.free_list[0] = (off + need, sz - need)
        ptr = self.base + off
        self.allocs[ptr] = (off, need)
        return ptr

    def reset(self) -> None:
        self.prev_carve = self.carve_sizes
        self.carve_sizes = []
        self.stale.update(self.allocs.keys())
        self.allocs.clear()
        self.free_list = [(0, self.size)]

    def free(self, ptr: int) -> bool:
        """Absorb without recycling: the range stays consumed until the
        next reset (bump purity beats intra-step reuse)."""
        if ptr in self.stale:
            self.stale.discard(ptr)
            return True
        if ptr in self.allocs:
            self.allocs.pop(ptr)
            return True
        return False


def _stabilize_tick(seq) -> None:
    """B1 lifecycle per sequence: runs 1-2 pass through (prefill +
    warmup), run 3 measures the transient high-water, runs 4+ carve
    every allocation from the sequence's recycled StepSlab. The active
    slab is switched at each run start (runs are synchronous, so the
    slab in force during a run is always the running sequence's)."""
    # Bank a pending measurement into ITS owner first (ticks of OTHER
    # sequences must never clobber the global measuring window — that
    # sized the decode slab from the sampler's tiny window and starved
    # it by 627 MB).
    if getattr(STATE, "measure_owner", None) is not None:
        STATE.measure_owner.__dict__["_stab_total"] = STATE.total_measured
        STATE.measure_owner = None
        STATE.measuring = False

    n = seq.__dict__.get("_stab_run", 0) + 1
    seq.__dict__["_stab_run"] = n
    if n <= 2:
        STATE.stab_slab = None
        return
    if n == 3:
        STATE.stab_slab = None
        STATE.measuring = True
        STATE.live = STATE.high_water = 0
        STATE.measured.clear()
        STATE.total_measured = 0
        STATE.measure_owner = seq
        return
    if "_stab_total" in seq.__dict__ and \
            seq.__dict__.get("_stab_slab") is None:
        # PURE-BUMP sizing: the slab must hold the step's TOTAL carved
        # bytes (no intra-step reuse), not the live high-water.
        need = int(seq.__dict__.pop("_stab_total") * 1.25) + (64 << 20)
        dev = None
        try:
            dev = DeviceAllocator.get_device()
        except Exception:
            pass
        seq.__dict__["_stab_slab"] = StepSlab(need, dev)
        comp = str((seq.dag or {}).get("component_name", "?"))
        print(f"[StepSlab] {comp}: stabilizing decode allocations "
              f"({need / 1e6:.0f} MB slab)", flush=True)
    slab = seq.__dict__.get("_stab_slab")
    if slab is not None:
        if not seq.__dict__.get("_stab_served_printed") and \
                seq.__dict__["_stab_run"] == 6:
            comp = str((seq.dag or {}).get("component_name", "?"))
            n_ops = len((seq.dag or {}).get("ops", {}))
            print(f"[StepSlab] {comp} (dag {n_ops} ops, seq id "
                  f"{id(seq) & 0xffff:04x}): served "
                  f"{len(slab.allocs) + len(slab.stale)} allocations "
                  f"last step (live {len(slab.allocs)}, "
                  f"stale {len(slab.stale)}, shortfall {slab.shortfall})",
                  flush=True)
            a, b = slab.prev_carve or [], slab.carve_sizes
            if a and b:
                if len(a) != len(b):
                    print(f"[StepSlab] {comp}: carve COUNT varies "
                          f"({len(a)} vs {len(b)})", flush=True)
                else:
                    div = [i for i, (x, y) in enumerate(zip(a, b))
                           if x != y]
                    if div:
                        i = div[0]
                        print(f"[StepSlab] {comp}: carve sizes diverge "
                              f"at {len(div)} indices; first at #{i}: "
                              f"{a[max(0,i-2):i+3]} vs "
                              f"{b[max(0,i-2):i+3]}", flush=True)
                    else:
                        print(f"[StepSlab] {comp}: carve size sequence "
                              f"IDENTICAL ({len(a)} allocs)", flush=True)
            seq.__dict__["_stab_served_printed"] = True
        slab.reset()
    STATE.stab_slab = slab


# ---------------------------------------------------------------------------
# Seams — installed once; routed by the module-level state below
# ---------------------------------------------------------------------------


class _State:
    def __init__(self) -> None:
        self.measuring = False
        self.recording = False
        self.records: List[Tuple[str, tuple]] = []
        self.broken: Optional[str] = None
        self.slab: Optional[SlabAllocator] = None
        # One slab PER DEVICE. `malloc_cuda` routed every allocation to the
        # single `slab` regardless of the requested `dev_idx`, so a
        # multi-device run would be served device 0's memory for a device 1
        # tensor — the second reason multi-device was locked out.
        self.slabs: Dict[int, SlabAllocator] = {}
        self.rec_dev: Optional[int] = None
        # Per-device measurement: one high-water mark cannot size slabs for a
        # run allocating on several cards without sizing each to the SUM.
        self.live_by_dev: Dict[int, int] = {}
        self.high_water_by_dev: Dict[int, int] = {}
        self.measured_dev: Dict[int, int] = {}
        # P-REPLAY-KV-DECODE step A: tuple census (pure observation —
        # per-run launch-tuple capture + consecutive-run diff; no
        # eligibility gates, no replay ever). NBX_REPLAY_TUPLE_CENSUS=1.
        self.census_current: Optional[List[tuple]] = None
        # B1: the running sequence's recycled StepSlab (or None) —
        # switched at each run start by _stabilize_tick under
        # NBX_REPLAY_KV_DECODE=1.
        self.stab_slab: Optional["StepSlab"] = None
        # Census attribution: the seq whose run the current capture
        # belongs to (armed at its tick; banked at the NEXT tick).
        self.census_owner = None
        # Measurement attribution (same off-by-one shape): the seq
        # whose run the measuring window covers.
        self.measure_owner = None
        # measuring counters (aligned bytes)
        self.live = 0
        self.high_water = 0
        self.measured: Dict[int, int] = {}  # ptr -> aligned size
        self.total_measured = 0             # cumulative (never decremented)

    def note_device(self) -> None:
        """Emit a device action when the current card differs from the last.

        Called before each recorded action rather than by wrapping
        `set_device`, because what has to be reproduced is the device each
        ACTION ran on, not every call that changed it.
        """
        try:
            dev = DeviceAllocator.get_device()
        except Exception:                                   # pragma: no cover
            return
        if dev != self.rec_dev:
            self.rec_dev = dev
            self.records.append((_SETDEV, (int(dev),)))

    def break_plan(self, reason: str) -> None:
        if self.broken is None:
            site = ""
            for fr in reversed(traceback.extract_stack()[:-2]):
                if "neurobrix" in fr.filename and "replay" not in fr.filename:
                    site = f" at {os.path.basename(fr.filename)}:{fr.lineno}"
                    break
            self.broken = reason + site


STATE = _State()

# Sequences that hold recorded plans (weak registry: the dict dies with
# the sequence, the SLABS do not — `retire_sequence_plans` is the release
# path an executor calls when it drops a sequence). The KV cache reaches
# the plans through `drop_plans_by_contribution` when it replaces a
# layer's buffers for a longer request — every plan recorded against the
# old addresses must go (D-SERVE-WARM-KV-GROWTH-ASYMMETRY: "replay-plan
# invalidation on growth"): a plan that survived the swap would replay
# kernel tuples, memcpys and captured graphs against freed or re-served
# memory. The signature LAYOUT is owned here (`signature()`): the owner
# contributions sit in one tuple at `sig[-2]`; callers match on a
# contribution, never on a position of the full key.
import weakref as _weakref
_PLAN_SEQS: "_weakref.WeakSet" = _weakref.WeakSet()


def _plan_of(state):
    """The FrozenPlan behind a plans-dict state, if it holds resources:
    a frozen plan, or a pending ("VERIFY", plan) whose slab was already
    allocated and whose verify pass has not run yet."""
    if isinstance(state, FrozenPlan):
        return state
    if isinstance(state, tuple) and len(state) == 2 and state[0] == "VERIFY" \
            and isinstance(state[1], FrozenPlan):
        return state[1]
    return None


def _drop_plan(plans: dict, sig) -> None:
    plan = _plan_of(plans.pop(sig, None))
    if plan is not None:
        plan.slab.retire()   # deferred: frees once carved tensors die
        if plan.graph is not None:
            try:
                plan.graph.destroy()
            finally:
                plan.graph = None


def _contributions(sig):
    if isinstance(sig, tuple) and len(sig) >= 2 and isinstance(sig[-2], tuple):
        return sig[-2]
    return ()


def drop_plans_by_contribution(pred) -> int:
    """Retire every recorded plan one of whose owner contributions
    satisfies `pred(contribution)` — frozen plans and pending VERIFY
    plans (slab retired, captured graph destroyed), plus the measuring /
    unreplayable states behind the same keys. Returns the number of keys
    dropped. Called by `TritonKVCache` at a buffer replacement with a
    predicate scoped to THAT cache's previous generation."""
    dropped = 0
    for seq in list(_PLAN_SEQS):
        plans = seq.__dict__.get("_replay_plans")
        if not plans:
            continue
        for sig in [k for k in plans if any(pred(c) for c in _contributions(k))]:
            _drop_plan(plans, sig)
            dropped += 1
    return dropped


def retire_sequence_plans(seq) -> int:
    """Release every plan a sequence holds (its slabs and captured
    graphs) — the executor's teardown path when it drops the sequence.
    The weak registry alone does not free them: a retired-less slab
    stays in `_ACTIVE_SLABS` with its base allocation for the process."""
    plans = seq.__dict__.get("_replay_plans")
    if not plans:
        return 0
    n = 0
    for sig in list(plans):
        _drop_plan(plans, sig)
        n += 1
    _PLAN_SEQS.discard(seq)
    return n
_INSTALLED = False
_ORIG_MALLOC = DeviceAllocator.malloc_cuda
_ORIG_FREE = DeviceAllocator.free_cuda
# Slabs with outstanding carved allocations — the free seam consults
# this PERMANENTLY (not only while recording): a slab-carved tensor may
# die on any later run when its arena slot is overwritten, and its
# finalizer must return the range to the slab, never cudaFree a
# slab-interior pointer.
_ACTIVE_SLABS: List["SlabAllocator"] = []


def _install_seams() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from triton.compiler.compiler import CompiledKernel

    orig_run_prop = CompiledKernel.run

    def _run_prop(self):  # noqa: ANN001
        raw = orig_run_prop.fget(self)
        if STATE.census_current is not None and not STATE.recording:
            def census_run(g0, g1, g2, stream, function, packed_metadata,
                           launch_md, enter_hook, exit_hook, *vals):
                flat = tuple(
                    int(v.data_ptr()) if hasattr(v, "data_ptr") else v
                    for v in vals)
                STATE.census_current.append(
                    (getattr(self, "name", "?"), g0, g1, g2, flat))
                return raw(g0, g1, g2, stream, function, packed_metadata,
                           launch_md, enter_hook, exit_hook, *vals)
            return census_run
        if not STATE.recording:
            return raw

        def recording_run(g0, g1, g2, stream, function, packed_metadata,
                          launch_md, enter_hook, exit_hook, *vals):
            flat = tuple(
                int(v.data_ptr()) if hasattr(v, "data_ptr") else v
                for v in vals)
           
            STATE.note_device()
            STATE.records.append(
                (_KERNEL, (self, g0, g1, g2, stream, flat)))
            return raw(g0, g1, g2, stream, function, packed_metadata,
                       launch_md, enter_hook, exit_hook, *vals)

        return recording_run

    CompiledKernel.run = property(_run_prop)

    def malloc_cuda(nbytes: int, dev_idx: Optional[int] = None) -> int:
        if STATE.recording and (STATE.slab is not None or STATE.slabs):
            # Route to the slab of the REQUESTED device. Serving a device-1
            # tensor from device 0's slab yields a pointer into the wrong
            # card's memory — a wrong result or an illegal access, never a
            # clear error.
            _dev = dev_idx if dev_idx is not None else DeviceAllocator.get_device()
            _slab = STATE.slabs.get(_dev, STATE.slab)
            if _slab is not None:
                ptr = _slab.malloc(nbytes)
                if ptr:
                    return ptr
                STATE.break_plan(
                    f"slab exhausted on device {_dev} (short {_slab.shortfall} B)")
            else:
                STATE.break_plan(f"no slab recorded for device {_dev}")
            return _ORIG_MALLOC(nbytes, dev_idx)
        if STATE.stab_slab is not None:
            ptr = STATE.stab_slab.malloc(nbytes)
            if ptr:
                return ptr
            # Graceful degradation: an over-slab allocation falls back
            # to the heap (address stability degrades for that tensor
            # only; the census quantifies the residue).
        ptr = _ORIG_MALLOC(nbytes, dev_idx)
        if STATE.measuring and ptr:
            need = (int(nbytes) + _ALIGN - 1) // _ALIGN * _ALIGN
            STATE.measured[ptr] = need
            STATE.total_measured += need
            STATE.live += need
            if STATE.live > STATE.high_water:
                STATE.high_water = STATE.live
            _d = dev_idx if dev_idx is not None else DeviceAllocator.get_device()
            STATE.measured_dev[ptr] = _d
            STATE.live_by_dev[_d] = STATE.live_by_dev.get(_d, 0) + need
            if STATE.live_by_dev[_d] > STATE.high_water_by_dev.get(_d, 0):
                STATE.high_water_by_dev[_d] = STATE.live_by_dev[_d]
        return ptr

    def free_cuda(ptr: int):
        for slab in _ACTIVE_SLABS:
            if slab.free(ptr):
                return  # returned to its slab (maybe releasing it)
        if STATE.measuring:
            need = STATE.measured.pop(ptr, 0)
            STATE.live -= need
            _d = STATE.measured_dev.pop(ptr, None)
            if _d is not None:
                STATE.live_by_dev[_d] = STATE.live_by_dev.get(_d, 0) - need
        return _ORIG_FREE(ptr)

    DeviceAllocator.malloc_cuda = staticmethod(malloc_cuda)
    DeviceAllocator.free_cuda = staticmethod(free_cuda)

    orig_memcpy = DeviceAllocator.memcpy

    def memcpy(dst, src, nbytes, kind: int = 3, *a, **kw):
        if STATE.recording:
            if kind == 2:
                STATE.break_plan("D2H memcpy during recording")
            elif kind == 1:
                import ctypes
                snap = ctypes.string_at(int(src), int(nbytes))
                STATE.note_device()
                STATE.records.append((_H2D, (int(dst), snap)))
            else:
                STATE.note_device()
                STATE.records.append((_MEMCPY, (int(dst), int(src),
                                                int(nbytes), kind)))
        return orig_memcpy(dst, src, nbytes, kind, *a, **kw)

    DeviceAllocator.memcpy = staticmethod(memcpy)

    orig_memset = DeviceAllocator.memset_cuda

    def memset_cuda(ptr, value, nbytes):
        if STATE.recording:
            STATE.note_device()
            STATE.records.append((_MEMSET, (int(ptr), int(value),
                                            int(nbytes))))
        return orig_memset(ptr, value, nbytes)

    DeviceAllocator.memset_cuda = staticmethod(memset_cuda)

    orig_item = NBXTensor.item

    def item(self):
        if STATE.recording:
            STATE.break_plan("NBXTensor.item() during recording")
        return orig_item(self)

    NBXTensor.item = item


# ---------------------------------------------------------------------------
# Frozen plan
# ---------------------------------------------------------------------------


class FrozenPlan:
    __slots__ = ("records", "slab", "arena_snapshot", "frozen_inputs",
                 "launches", "actions", "output_slots", "scan_limit",
                 "graph", "graph_disabled")

    def __init__(self, records: List[Tuple[str, tuple]],
                 slab: SlabAllocator,
                 arena_list: List[Optional[NBXTensor]],
                 input_slots: Tuple[int, ...],
                 scan_limit: Optional[int] = None) -> None:
        self.records = records
        self.slab = slab  # strong ref pins the slab for the plan's life
        self.arena_snapshot = list(arena_list)
        self.frozen_inputs: Dict[int, NBXTensor] = {}
        for s in input_slots:
            t = arena_list[s]
            if t is not None:
                self.frozen_inputs[s] = t
        self.launches = sum(1 for r in records if r[0] == _KERNEL)
        self.actions = len(records)
        self.output_slots: Tuple[int, ...] = ()
        # Scan bound: weights + inputs (constants are weight-region
        # slots). Intermediates are NEVER copied — the DAG produces
        # every intermediate before consuming it, so the replayed
        # launches rewrite the frozen buffers from the frozen inputs;
        # copying them was both wasted work (tens of thousands of slots
        # on an LLM decode graph) and a dangling-source hazard: after a
        # verify pass's normal re-run, an intermediate slot can hold a
        # VIEW whose base died with the next rebind (measured: KV-decode
        # gate v2, slot 18939, GQA view, cudaMemcpy rc=1 on the freed
        # source).
        self.scan_limit = (len(arena_list) if scan_limit is None
                          else scan_limit)
        # Phase 4b: lazily captured CUDA graph of the action list
        # (opt-in NBX_REPLAY_GRAPH=1, A/B-gated adoption). A capture
        # failure disables the graph for this bucket only — direct
        # replay stays the correct path.
        self.graph = None
        self.graph_disabled = False

    def replay(self, arena) -> None:
        # Freeze-and-copy over rebound WEIGHT/CONSTANT/INPUT slots:
        # anything rebound between steps with a NEW object (bind_inputs
        # slots, but also per-step re-sliced seq-dependent constants —
        # probe 4 caught a strided_copy at replay action 6 reading a
        # freed previous-step constant) gets its bytes copied into the
        # frozen buffer so every recorded pointer stays live and
        # current. Identity scan cost is ~µs; identical objects skip.
        _scan_copied = 0
        # C-level pairing (zip + islice over the raw slot list) — the
        # indexed per-slot form cost ~19k Arena.__getitem__ per decode
        # token on the 30B row (2026-08-23 host profile).
        from itertools import islice
        for slot, (frozen, new) in enumerate(
                islice(zip(self.arena_snapshot, arena._slots),
                       self.scan_limit)):
            if frozen is None or new is None or new is frozen:
                continue
            if (tuple(new.shape) != tuple(frozen.shape)
                    or new.nbx_dtype != frozen.nbx_dtype):
                raise RuntimeError(
                    "ZERO FALLBACK: replay slot mismatch at slot "
                    f"{slot}: {tuple(new.shape)}/{new.nbx_dtype} vs "
                    f"frozen {tuple(frozen.shape)}/{frozen.nbx_dtype} "
                    "— the bucket signature should have caught this")
            _scan_copied += 1
            try:
                DeviceAllocator.memcpy(frozen.data_ptr(), new.data_ptr(),
                                       frozen._nbytes, 3)
            except Exception as e:
                raise RuntimeError(
                    f"replay input-copy failed at slot {slot}: shape "
                    f"{tuple(frozen.shape)} dtype {frozen.nbx_dtype} "
                    f"nbytes {frozen._nbytes} frozen_dev="
                    f"{getattr(frozen, '_device_idx', '?')} new_dev="
                    f"{getattr(new, '_device_idx', '?')} frozen_ptr="
                    f"{frozen.data_ptr():#x} new_ptr={new.data_ptr():#x}"
                ) from e
        if os.environ.get("NBX_REPLAY_GRAPH_DUMP") == "1":
            import ctypes as _cs
            vals = []
            for s, ft in self.frozen_inputs.items():
                if ft._nbytes <= 16:
                    b = (_cs.c_int64 * (ft._nbytes // 8))()
                    DeviceAllocator.memcpy(_cs.addressof(b),
                                           ft.data_ptr(), ft._nbytes, 2)
                    vals.append((s, list(b)))
            print(f"[Replay][SCAN] copied={_scan_copied} "
                  f"small_inputs={vals}", flush=True)
        if (os.environ.get("NBX_REPLAY_GRAPH") == "1"
                and not self.graph_disabled):
            # Full-list capture. The launch is bracketed by device-wide
            # syncs (CapturedPlan.launch): the input-scan copies are
            # async legacy-stream work a NON_BLOCKING graph stream
            # would NOT wait for — the 2026-08-17 root cause of the
            # stale-input token doubling (all intermediate "breaking
            # node" bisection boundaries were timing artifacts of that
            # race, masked by diagnostic D2H syncs).
            # NBX_REPLAY_GRAPH_CUT=N remains a bisection instrument
            # (prefix in-graph, remainder direct — order preserved).
            cut = int(os.environ.get("NBX_REPLAY_GRAPH_CUT", "0") or 0)
            recs = self.records[:cut] if cut > 0 else self.records
            if cut <= 0:
                cut = len(self.records)
            try:
                if self.graph is None:
                    from neurobrix.triton.replay_graph import CapturedPlan
                    if os.environ.get("NBX_REPLAY_GRAPH_DUMP") == "1":
                        from collections import Counter as _Ctr
                        seqd = "".join(t for t, _ in recs[:60])
                        names = [getattr(r[0], "name", "?")[:28]
                                 for t, r in recs[:12] if t == _KERNEL]
                        cnt = _Ctr(t for t, _ in self.records)
                        sizes = sorted({len(r[1]) for t, r in self.records
                                        if t == _H2D})
                        print(f"[Replay][GRAPH] action tags[:60]={seqd} "
                              f"counts={dict(cnt)} h2d_sizes={sizes[:8]} "
                              f"first kernels={names}", flush=True)
                        tail = [(t, getattr(r[0], "name", "?")[:24]
                                 if t == _KERNEL else
                                 (r[2] if t in (_MEMCPY,) else len(r[1])
                                  if t == _H2D else r))
                                for t, r in self.records[-16:]]
                        print(f"[Replay][GRAPH] tail16={tail}", flush=True)
                    self.graph = CapturedPlan(
                        recs, getattr(self.slab, "dev_idx", None))
                    print(f"[Replay][GRAPH] bucket captured: "
                          f"{self.graph.launches} launches as one CUDA "
                          f"graph ({cut}/{len(self.records)} actions "
                          f"in-graph)")
                self.graph.launch()
            except Exception as e:
                if getattr(e, "executed", False):
                    # The graph's work was already submitted — device
                    # state (KV counters) has advanced; re-running the
                    # actions would double-advance it. Crash loudly.
                    raise RuntimeError(
                        "replay graph post-execution failure — state "
                        "advanced, no safe fallback") from e
                # Capture / pre-launch failure: nothing executed.
                # Loud per-bucket refusal — direct replay is the
                # correct path by construction (llama.cpp guard class).
                print(f"[Replay][GRAPH] refusal ({e}) — bucket keeps "
                      f"direct replay")
                self.graph_disabled = True
                if self.graph is not None:
                    try:
                        self.graph.destroy()
                    finally:
                        self.graph = None
                self._direct_actions()
            else:
                # Bisection tail (order-preserving): runs OUTSIDE the
                # refusal try — a tail failure is a direct-replay
                # failure (loud RuntimeError), never a trigger for
                # re-running the already-executed graph prefix.
                if cut < len(self.records):
                    self._direct_actions(self.records[cut:])
        else:
            self._direct_actions()
        arena.restore_from(self.arena_snapshot)
        # Contract clause 2 (inter-replay overwrite), enforced HERE so
        # no flow needs to cooperate: callers may legally retain output
        # objects across calls (VibeVoice chunk accumulation appends by
        # reference — battery adjudication 2026-08-13: every appended
        # chunk was THE SAME frozen tensor, rewritten by the next
        # replay). Each replay therefore returns FRESH copies of the
        # graph outputs; the frozen buffers stay plan-internal.
        for slot in self.output_slots:
            frozen = self.arena_snapshot[slot]
            if frozen is None:
                continue
            fresh = NBXTensor.empty_like(frozen)
            DeviceAllocator.memcpy(fresh.data_ptr(), frozen.data_ptr(),
                                   frozen._nbytes, 3)
            arena[slot] = fresh

    def _direct_actions(self, records=None) -> None:
        """Phase 4a direct C-launcher replay of the action list."""
        if records is None:
            records = self.records
        for idx, (tag, rec) in enumerate(records):
            try:
                if tag == _SETDEV:
                    # Triton keeps its OWN current device, so setting the
                    # runtime's alone would still launch on the wrong card.
                    DeviceAllocator.set_device(rec[0])
                    DeviceAllocator.ensure_triton_device(rec[0])
                elif tag == _KERNEL:
                    kernel, g0, g1, g2, stream, flat = rec
                    kernel.run(g0, g1, g2, stream, kernel.function,
                               kernel.packed_metadata, None, None, None,
                               *flat)
                elif tag == _MEMCPY:
                    DeviceAllocator.memcpy(*rec)
                elif tag == _H2D:
                    import ctypes
                    dst, snap = rec
                    buf = (ctypes.c_char * len(snap)).from_buffer_copy(snap)
                    DeviceAllocator.memcpy(dst, ctypes.addressof(buf),
                                           len(snap), 1)
                else:
                    DeviceAllocator.memset_cuda(*rec)
            except BaseException as e:
                name = (getattr(rec[0], "name", "?")
                        if tag == _KERNEL else tag)
                raise RuntimeError(
                    f"replay action {idx}/{len(self.records)} "
                    f"({name}) failed: {e}") from e


# ---------------------------------------------------------------------------
# Orchestration — called from TritonSequence.run
# ---------------------------------------------------------------------------


def _input_slot_range(seq) -> range:
    """Arena layout is [weights | inputs | intermediates]."""
    return range(seq._num_weights, seq._num_weights + seq._num_inputs)


def _output_hashes(seq) -> Dict[str, str]:
    """sha256 of every graph-output tensor's device bytes (D2H once —
    verify runs once per bucket per process)."""
    import ctypes
    import hashlib
    out: Dict[str, str] = {}
    for tid in seq.dag.get("output_tensor_ids") or []:
        slot = seq._tid_to_slot.get(tid)
        if slot is None:
            continue
        t = seq._arena[slot]
        if t is None:
            out[tid] = "none"
            continue
        buf = (ctypes.c_char * t._nbytes)()
        DeviceAllocator.memcpy(ctypes.addressof(buf), t.data_ptr(),
                               t._nbytes, 2)
        out[tid] = hashlib.sha256(bytes(buf)).hexdigest()
    return out


def signature(seq) -> Optional[tuple]:
    if getattr(seq, "_is_multi_device", False):
        return None
    # Stage-driven sequences (core/flow/stages/ calling convention —
    # the documented R33-exception path) live outside the standard
    # flow contract: five falsified byte-divergence mechanisms on the
    # VibeVoice tokenizers (2026-08-13 adjudication log) put them
    # behind the capability gate until the residual is root-caused.
    if getattr(seq, "_replay_ineligible", False):
        return None
    # STATEFUL interceptors (KV-cache attention class) advance internal
    # state per call while the sequence's input signature stays
    # constant — a recorded step would bake one cache length and replay
    # it forever (battery adjudication 2026-08-13: tinyllama_triton
    # text diff + vibevoice_triton wav diff, both KV-decode legs).
    # Registration contract (P-REPLAY-KV-DECODE): interceptors whose
    # owner implements (replay_signature, replay_advance,
    # replay_restore) contribute their state to the bucket signature
    # instead; anything unregistered refuses as before.
    owners: List[Any] = []
    contrib: List[Any] = []
    if seq._op_uid_interceptors or seq._op_interceptors:
        by_owner: Dict[int, Tuple[Any, List[Any]]] = {}
        for func in list(seq._op_interceptors.values()) + \
                list(seq._op_uid_interceptors.values()):
            owner = getattr(func, "__self__", None)
            if owner is None or not (
                    hasattr(owner, "replay_signature")
                    and hasattr(owner, "replay_advance")
                    and hasattr(owner, "replay_restore")):
                return None
            by_owner.setdefault(id(owner), (owner, []))[1].append(func)
        for owner, funcs in by_owner.values():
            s = owner.replay_signature(funcs)
            if s is None:
                return None
            contrib.append(s)
            owners.append(owner)
    seq.__dict__["_replay_state_owners"] = owners
    # v1 eligibility: in-graph RNG ops draw fresh values per step from
    # host-advanced generator state baked into launch scalars — a
    # recorded draw would replay ONE step's noise forever (battery
    # adjudication 2026-08-13: vibevoice_triton wav diff, DDPM class).
    nondet = getattr(seq, "_replay_has_nondet", None)
    if nondet is None:
        _NONDET = ("rand", "randn", "multinomial", "bernoulli",
                   "normal", "uniform", "exponential", "randint")
        nondet = any(
            any(tok in op.op_type for tok in _NONDET)
            for op in seq._ops)
        seq._replay_has_nondet = nondet
    if nondet:
        return None
    parts: List[Any] = []
    for s in _input_slot_range(seq):
        t = seq._arena[s]
        if t is None:
            parts.append((s, None))
        else:
            parts.append((s, tuple(t.shape), int(t.nbx_dtype)))
    resolver = getattr(seq, "_symbol_resolver", None)
    resolved = getattr(resolver, "resolved", None) if resolver else None
    if isinstance(resolved, dict):
        parts.append(tuple(sorted(
            (k, v) for k, v in resolved.items()
            if isinstance(v, (int, float, str)))))
    parts.append(tuple(sorted(seq._op_uid_interceptors)))
    parts.append(tuple(sorted(seq._op_interceptors)))
    # Registration-contract state (e.g. the KV decode bucket): a state
    # change — bucket boundary crossed — lands the run in a NEW bucket
    # (measure/record/verify again) instead of replaying stale extents.
    parts.append(tuple(contrib))
    parts.append(bool(getattr(seq, "_activations_fp16_safe", False)))
    sig = tuple(parts)
    if os.environ.get("NBX_REPLAY_SIG_DIAG") == "1":
        prev = seq.__dict__.get("_sig_diag_prev")
        if prev is not None and prev != sig and len(prev) == len(sig):
            for i, (a, b) in enumerate(zip(prev, sig)):
                if a != b:
                    print(f"[Replay][SIG] part {i} changed: "
                          f"{repr(a)[:200]} -> {repr(b)[:200]}",
                          flush=True)
            for o in owners:
                print(f"[Replay][SIG] owner id={id(o) & 0xffffff:06x} "
                      f"cache_len={o.get_cache_len()}", flush=True)
        seq.__dict__["_sig_diag_prev"] = sig
    return sig


_AT_SEEDED = False


def _seed_autotune_once() -> None:
    """E2-full: seed the sanctioned Autotuner caches from the
    arch-keyed artifact once per process — a seeded key means run()
    never benches (kills first-request tuning cost AND the tuning
    timing-variance surface on recorded shapes)."""
    global _AT_SEEDED
    if _AT_SEEDED:
        return
    _AT_SEEDED = True
    try:
        from neurobrix.triton.autotune_cache import seed as _at_seed
        n = _at_seed()
        if n:
            print(f"[Replay] autotune artifact seeded: {n} configs")
    except (OSError, ValueError):
        pass  # artifact I/O is an optimization, never a failure source
    except Exception as e:  # API drift must stay observable, once
        print(f"[Replay] autotune seed unavailable ({type(e).__name__}: {e})")


def _census_tick(seq) -> None:
    """Tuple census (P-REPLAY-KV-DECODE step A): close the previous
    run's capture, diff it against the run before, print the varying
    surface ONCE per sequence, arm the next capture. Enabled by
    NBX_TRITON_REPLAY=1 + NBX_REPLAY_TUPLE_CENSUS=1 together (the
    sequence only consults the replayer under the first flag; the
    second short-circuits every plan path — observation only)."""
    # Bank the finished capture into the seq that RAN it (the owner
    # armed at the previous tick) — banking into the ticking seq
    # misattributed every window once several sequences interleaved
    # (decode / lm_head / sampler) and made every tuple "vary".
    if STATE.census_current is not None and STATE.census_owner is not None:
        STATE.census_owner.__dict__.setdefault("_census_runs", []).append(
            STATE.census_current)
        _census_report(STATE.census_owner)
    STATE.census_current = []
    STATE.census_owner = seq
    return


def _census_report(seq) -> None:
    prev_runs = seq.__dict__.get("_census_runs", [])
    # Compare runs 3 vs 4 (two STEADY decode steps): run 1 is the
    # prefill (different shape class by design — vLLM never captures
    # it either) and run 2 is the first decode (warmup/autotune).
    # Under B1 stabilization, run 3 measures (heap) and run 4 is the
    # FIRST slab run — compare runs 5 vs 6 (two stabilized steps).
    min_runs = 6 if os.environ.get("NBX_REPLAY_KV_DECODE") == "1" else 4
    if len(prev_runs) < min_runs or seq.__dict__.get("_census_printed"):
        return
    a, b = prev_runs[-2], prev_runs[-1]
    comp = str((seq.dag or {}).get("component_name", "?"))
    if len(a) != len(b):
        print(f"[TupleCensus] {comp}: launch COUNT varies "
              f"({len(a)} vs {len(b)}) — structural per-step change",
              flush=True)
        seq.__dict__["_census_printed"] = True
        return
    varying = []
    for i, (la, lb) in enumerate(zip(a, b)):
        if la == lb:
            continue
        name_a, g0a, g1a, g2a, fa = la
        _, g0b, g1b, g2b, fb = lb
        grid_diff = (g0a, g1a, g2a) != (g0b, g1b, g2b)
        arg_idx = [j for j, (x, y) in enumerate(zip(fa, fb)) if x != y]
        varying.append((i, name_a, grid_diff, arg_idx))
    print(f"[TupleCensus] {comp}: {len(a)} launches, "
          f"{len(varying)} vary between consecutive runs", flush=True)
    from collections import Counter
    by_kernel = Counter((v[1], v[2], tuple(v[3])) for v in varying)
    for (kname, gdiff, args), cnt in by_kernel.most_common(12):
        print(f"  {cnt:5d}x {kname}  grid_varies={gdiff}  "
              f"varying_arg_positions={list(args)}", flush=True)
    seq.__dict__["_census_printed"] = True


def _mem_diag(tag: str) -> None:
    """NBX_REPLAY_MEM_DIAG=1: print the allocator's live-tracked bytes
    at replay state transitions (retention forensics)."""
    if os.environ.get("NBX_REPLAY_MEM_DIAG") != "1":
        return
    total = sum(DeviceAllocator._cuda_live_bytes.values())
    print(f"[Replay][MEM] {tag}: live_tracked={total/1e6:.0f}MB",
          flush=True)


def would_replay(seq, pre_op_callback) -> bool:
    """Side-effect-free: would `maybe_run` take the frozen-plan path?

    Exists so the caller can SKIP the per-step setup that only serves
    op-by-op execution — `bind_weights` and the `compute_op_devices` it
    ends with — when this step will replay a frozen plan anyway. The
    host profile of the captured decode step put that setup at the heart
    of the ~65 ms/token of host time: the replay executes BAKED
    pointers, so re-binding weights every token was pure redundancy *by
    the replay's own trust model* — if the weights had actually moved,
    the frozen launches would already be invalid.

    Deliberately conservative: every path this predicate cannot vouch
    for (census mode, pre_op callbacks — the zero3 class —, multi-device,
    stage-driven, unregistered stateful interceptors, no frozen plan
    yet) returns False and the caller binds exactly as before. And the
    skip is guarded end-to-end: the caller parks the weights on the
    sequence, and if `maybe_run` then declines for ANY reason, `run()`
    performs the parked bind before executing a single op.
    """
    if os.environ.get("NBX_REPLAY_TUPLE_CENSUS") == "1":
        return False
    if not ENABLED or pre_op_callback is not None:
        return False
    sig = signature(seq)
    if sig is None:
        return False
    only = os.environ.get("NBX_REPLAY_ONLY")
    if only:
        comp = str((seq.dag or {}).get("component_name", "?"))
        if comp not in only.split(","):
            return False
    state = seq.__dict__.get("_replay_plans", {}).get(sig)
    return isinstance(state, FrozenPlan)


def maybe_run(seq, skip_kills: bool, pre_op_callback) -> bool:
    """Replay fast path. True = this run was fully handled.

    Mode composition: NBX_REPLAY_TUPLE_CENSUS short-circuits every plan
    path (pure observation; with NBX_REPLAY_KV_DECODE it also runs the
    B1 StepSlab stabilizer so the census sees stabilized addresses).
    WITHOUT the census flag, NBX_REPLAY_KV_DECODE composes with the
    plan machine directly: the B1 stabilizer stays OFF (the recording
    slab is the address-pinning authority) and the interceptor
    registration contract in signature() decides eligibility."""
    if os.environ.get("NBX_REPLAY_TUPLE_CENSUS") == "1":
        _install_seams()
        if os.environ.get("NBX_REPLAY_KV_DECODE") == "1":
            _stabilize_tick(seq)
        _census_tick(seq)
        return False
    if not ENABLED or pre_op_callback is not None:
        return False
    _seed_autotune_once()
    sig = signature(seq)
    if sig is None:
        return False
    only = os.environ.get("NBX_REPLAY_ONLY")
    if only:
        comp = str((seq.dag or {}).get("component_name", "?"))
        if comp not in only.split(","):
            return False
    plans = seq.__dict__.setdefault("_replay_plans", {})
    _PLAN_SEQS.add(seq)
    state = plans.get(sig)

    owners = seq.__dict__.get("_replay_state_owners", ())

    if isinstance(state, FrozenPlan):
        if os.environ.get("NBX_REPLAY_VERIFY_EVERY") == "1":
            # Diagnostic mode: verify EVERY replay, name the first
            # diverging occurrence per bucket.
            state.replay(seq._arena)
            replayed = _output_hashes(seq)
            for o in owners:
                o.replay_restore()
            seq._run_single_device(skip_kills, None)
            normal = _output_hashes(seq)
            n = seq.__dict__.setdefault("_replay_occurrence", {})
            n[sig] = n.get(sig, 0) + 1
            if replayed != normal:
                print(f"[Replay][VERIFY_EVERY] DIVERGENCE at "
                      f"occurrence {n[sig]} of bucket "
                      f"({state.launches} launches)")
                if os.environ.get("NBX_REPLAY_GRAPH_DUMP") == "1":
                    for tid in (seq.dag.get("output_tensor_ids")
                                or [])[:1]:
                        slot = seq._tid_to_slot.get(tid)
                        frozen = state.arena_snapshot[slot] \
                            if slot is not None else None
                        if frozen is None:
                            continue
                        import ctypes as _c
                        import numpy as _np
                        nb = min(frozen._nbytes, 64)
                        rb = (_c.c_char * nb)()
                        DeviceAllocator.memcpy(_c.addressof(rb),
                                               frozen.data_ptr(), nb, 2)
                        fro = _np.frombuffer(bytes(rb), dtype=_np.float16)
                        cur = seq._arena[slot]
                        rb2 = (_c.c_char * nb)()
                        DeviceAllocator.memcpy(_c.addressof(rb2),
                                               cur.data_ptr(), nb, 2)
                        nor = _np.frombuffer(bytes(rb2),
                                             dtype=_np.float16)
                        print(f"[Replay][VERIFY_EVERY] out[:8] "
                              f"replayed(frozen)={fro[:8].tolist()} "
                              f"normal={nor[:8].tolist()}", flush=True)
            return True
        state.replay(seq._arena)
        for o in owners:
            o.replay_advance()
        n_r = seq.__dict__.setdefault("_replay_count", {})
        n_r[sig] = n_r.get(sig, 0) + 1
        if n_r[sig] % 50 == 1:
            _mem_diag(f"replay #{n_r[sig]}")
        if os.environ.get("NBX_REPLAY_GRAPH_DUMP") == "1" and owners:
            import ctypes as _c
            for o in owners:
                lay = next(iter(o.cache._layers.values()), None)
                if lay is None or lay.pos_counter is None:
                    continue
                b = (_c.c_int32 * 1)()
                DeviceAllocator.memcpy(_c.addressof(b),
                                       lay.pos_counter.data_ptr(), 4, 2)
                print(f"[Replay][GRAPH] post-replay dev_pos={b[0]} "
                      f"host_len={lay.current_len}", flush=True)
        return True
    if isinstance(state, tuple) and state[0] == "VERIFY":
        # verify-first-replay (universal guard, battery adjudication
        # 2026-08-13: a vibevoice stage-handler bucket carried
        # per-step state none of the static guards see). Replay, hash
        # the graph outputs, then run the NORMAL path on the same
        # inputs and compare: equal -> plan confirmed; different ->
        # plan discarded loudly. Either way this run's results come
        # from the NORMAL path (correct by construction). One doubled
        # step per bucket per process.
        plan = state[1]
        plan.replay(seq._arena)
        replayed = _output_hashes(seq)
        # Registration contract: rewind the replayed pass's
        # non-idempotent device writes (e.g. KV position counters) so
        # the normal re-run below operates on the SAME state — it then
        # advances host+device consistently itself.
        for o in owners:
            o.replay_restore()
        seq._run_single_device(skip_kills, None)
        normal = _output_hashes(seq)
        if replayed == normal:
            plans[sig] = plan
            print(f"[Replay] plan VERIFIED byte-equal "
                  f"({plan.launches} launches) — replaying this bucket")
            _mem_diag("post-verify")
        else:
            plan.slab.retire()
            if plan.graph is not None:
                # The verify pass may have captured the CUDA graph —
                # release exec/graph/staging/stream with the rejected
                # plan (gardien 2026-08-17: the drop leaked them).
                try:
                    plan.graph.destroy()
                finally:
                    plan.graph = None
            plans[sig] = "UNREPLAYABLE"
            print("[Replay] plan REJECTED at verify (outputs differ "
                  "from the normal path) — normal path keeps this "
                  "graph")
        return True
    if state == "UNREPLAYABLE":
        return False

    _install_seams()

    if state is None:
        cached = _load_slab_size(seq, sig)
        if cached:
            # Warm restart: size known — skip the measure pass, this
            # run becomes the pre-record warmup (autotune still needs
            # one eager pass per process).
            plans[sig] = ("MEASURED", int(cached), 1)
            seq._run_single_device(skip_kills, None)
            return True
        # MEASURE pass: normal run under the watermark counter.
        STATE.measuring = True
        STATE.live = 0
        STATE.high_water = 0
        STATE.measured = {}
        try:
            seq._run_single_device(skip_kills, None)
        finally:
            STATE.measuring = False
        plans[sig] = ("MEASURED", STATE.high_water, 1)
        _mem_diag("post-measure")
        return True

    # ("MEASURED", slab_bytes, attempt) — record this execution.
    _, slab_bytes, attempt = state
    if slab_bytes <= 0:
        plans[sig] = "UNREPLAYABLE"
        return False
    try:
        slab = SlabAllocator(slab_bytes, seq.device_idx)
    except Exception as e:
        print(f"[Replay] slab alloc failed ({slab_bytes/1e6:.0f} MB): "
              f"{e} — bucket keeps the normal path")
        plans[sig] = "UNREPLAYABLE"
        return False

    STATE.recording = True
    STATE.records = []
    STATE.broken = None
    STATE.slab = slab
    failed: Optional[BaseException] = None
    try:
        seq._run_single_device(skip_kills, None)
    except BaseException as e:  # noqa: BLE001 — recording must not corrupt user runs
        failed = e
    finally:
        STATE.recording = False
        STATE.slab = None
        records = STATE.records
        STATE.records = []

    if failed is not None or STATE.broken is not None:
        reason = STATE.broken or f"exception: {failed}"
        shortfall = slab.shortfall
        slab.retire()  # deferred: frees only after carved tensors die
        if failed is None and shortfall > 0 and attempt < 2:
            # Data-driven retry: the failed attempt measured the exact
            # extra bytes needed.
            plans[sig] = ("MEASURED", slab_bytes + shortfall, attempt + 1)
            print(f"[Replay] slab undersized by {shortfall/1e6:.0f} MB — "
                  f"will re-record at {(slab_bytes+shortfall)/1e6:.0f} MB")
            return True  # the recording run still executed correctly
        plans[sig] = "UNREPLAYABLE"
        print(f"[Replay] bucket UNREPLAYABLE ({reason}) — normal path "
              f"keeps this graph")
        if failed is not None:
            # The failed partial run may have left the arena mid-state:
            # re-execute normally from the top (ops are deterministic
            # and slot writes are idempotent across a rerun).
            seq._run_single_device(skip_kills, None)
        return True

    plan = FrozenPlan(records, slab, list(seq._arena),
                      tuple(_input_slot_range(seq)),
                      scan_limit=seq._num_weights + seq._num_inputs)
    plan.output_slots = tuple(
        s for s in (seq._tid_to_slot.get(tid)
                    for tid in seq.dag.get("output_tensor_ids") or [])
        if s is not None)
    plans[sig] = ("VERIFY", plan)
    # The RECORDING run's own outputs are slab-carved — the caller is
    # about to gather and may RETAIN them, and every later replay
    # rewrites the slab (battery adjudication 2026-08-13: VibeVoice's
    # chunk-2 audio, recorded-run output, was corrupted retroactively
    # by subsequent replays). Hand the caller fresh heap copies NOW
    # (freeze happens inside this run, before gather_outputs); the
    # slab-backed originals stay plan-internal via the snapshot.
    for slot in plan.output_slots:
        frozen = plan.arena_snapshot[slot]
        if frozen is None:
            continue
        fresh = NBXTensor.empty_like(frozen)
        DeviceAllocator.memcpy(fresh.data_ptr(), frozen.data_ptr(),
                               frozen._nbytes, 3)
        seq._arena[slot] = fresh
    _store_slab_size(seq, sig, slab.size)
    # E2-full: the recording pass just warmed the sanctioned autotuners
    # for every shape in this graph — persist the selected configs into
    # the arch-keyed artifact (seeded back on the next process's
    # install; immune to triton's source-hash invalidation).
    try:
        from neurobrix.triton.autotune_cache import capture as _at_capture
        _at_capture()
    except (OSError, ValueError):
        pass  # artifact I/O is an optimization, never a failure source
    except Exception as e:  # API drift must stay observable, once
        print(f"[Replay] autotune capture unavailable "
              f"({type(e).__name__}: {e})")
    comp = str((seq.dag or {}).get("component_name", "?"))
    print(f"[Replay] plan frozen[{comp}]: {plan.launches} launches "
          f"({plan.actions} actions), slab "
          f"{slab.size/1e6:.0f} MB, inputs {len(plan.frozen_inputs)}, "
          f"outputs {len(plan.output_slots)}")
    _mem_diag("post-record")
    return True
