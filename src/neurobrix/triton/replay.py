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
        # P-REPLAY-KV-DECODE step A: tuple census (pure observation —
        # per-run launch-tuple capture + consecutive-run diff; no
        # eligibility gates, no replay ever). NBX_REPLAY_TUPLE_CENSUS=1.
        self.census_current: Optional[List[tuple]] = None
        # measuring counters (aligned bytes)
        self.live = 0
        self.high_water = 0
        self.measured: Dict[int, int] = {}  # ptr -> aligned size

    def break_plan(self, reason: str) -> None:
        if self.broken is None:
            site = ""
            for fr in reversed(traceback.extract_stack()[:-2]):
                if "neurobrix" in fr.filename and "replay" not in fr.filename:
                    site = f" at {os.path.basename(fr.filename)}:{fr.lineno}"
                    break
            self.broken = reason + site


STATE = _State()
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
            STATE.records.append(
                (_KERNEL, (self, g0, g1, g2, stream, flat)))
            return raw(g0, g1, g2, stream, function, packed_metadata,
                       launch_md, enter_hook, exit_hook, *vals)

        return recording_run

    CompiledKernel.run = property(_run_prop)

    def malloc_cuda(nbytes: int, dev_idx: Optional[int] = None) -> int:
        if STATE.recording and STATE.slab is not None:
            ptr = STATE.slab.malloc(nbytes)
            if ptr:
                return ptr
            STATE.break_plan(
                f"slab exhausted (short {STATE.slab.shortfall} B)")
            return _ORIG_MALLOC(nbytes, dev_idx)
        ptr = _ORIG_MALLOC(nbytes, dev_idx)
        if STATE.measuring and ptr:
            need = (int(nbytes) + _ALIGN - 1) // _ALIGN * _ALIGN
            STATE.measured[ptr] = need
            STATE.live += need
            if STATE.live > STATE.high_water:
                STATE.high_water = STATE.live
        return ptr

    def free_cuda(ptr: int):
        for slab in _ACTIVE_SLABS:
            if slab.free(ptr):
                return  # returned to its slab (maybe releasing it)
        if STATE.measuring:
            need = STATE.measured.pop(ptr, 0)
            STATE.live -= need
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
                STATE.records.append((_H2D, (int(dst), snap)))
            else:
                STATE.records.append((_MEMCPY, (int(dst), int(src),
                                                int(nbytes), kind)))
        return orig_memcpy(dst, src, nbytes, kind, *a, **kw)

    DeviceAllocator.memcpy = staticmethod(memcpy)

    orig_memset = DeviceAllocator.memset_cuda

    def memset_cuda(ptr, value, nbytes):
        if STATE.recording:
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
                 "launches", "actions", "output_slots")

    def __init__(self, records: List[Tuple[str, tuple]],
                 slab: SlabAllocator,
                 arena_list: List[Optional[NBXTensor]],
                 input_slots: Tuple[int, ...]) -> None:
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

    def replay(self, arena) -> None:
        # Freeze-and-copy over ALL slots, not only declared inputs:
        # anything rebound between steps with a NEW object (bind_inputs
        # slots, but also per-step re-sliced seq-dependent constants —
        # probe 4 caught a strided_copy at replay action 6 reading a
        # freed previous-step constant) gets its bytes copied into the
        # frozen buffer so every recorded pointer stays live and
        # current. Identity scan cost is ~µs; identical objects skip.
        for slot, frozen in enumerate(self.arena_snapshot):
            if frozen is None:
                continue
            new = arena[slot]
            if new is None or new is frozen:
                continue
            if (tuple(new.shape) != tuple(frozen.shape)
                    or new.nbx_dtype != frozen.nbx_dtype):
                raise RuntimeError(
                    "ZERO FALLBACK: replay slot mismatch at slot "
                    f"{slot}: {tuple(new.shape)}/{new.nbx_dtype} vs "
                    f"frozen {tuple(frozen.shape)}/{frozen.nbx_dtype} "
                    "— the bucket signature should have caught this")
            DeviceAllocator.memcpy(frozen.data_ptr(), new.data_ptr(),
                                   frozen._nbytes, 3)
        for idx, (tag, rec) in enumerate(self.records):
            try:
                if tag == _KERNEL:
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
        for i, t in enumerate(self.arena_snapshot):
            arena[i] = t
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
    # v1 eligibility: STATEFUL interceptors (KV-cache attention class)
    # advance internal state per call while the sequence's input
    # signature stays constant — a recorded step would bake one cache
    # length and replay it forever (battery adjudication 2026-08-13:
    # tinyllama_triton text diff + vibevoice_triton wav diff, both
    # KV-decode legs). Any registered interceptor ⇒ ineligible until
    # interceptor state enters the signature (named next increment).
    if seq._op_uid_interceptors or seq._op_interceptors:
        return None
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
    parts.append(bool(getattr(seq, "_activations_fp16_safe", False)))
    return tuple(parts)


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
    prev_runs = seq.__dict__.setdefault("_census_runs", [])
    if STATE.census_current is not None:
        prev_runs.append(STATE.census_current)
    STATE.census_current = []
    # Compare runs 3 vs 4 (two STEADY decode steps): run 1 is the
    # prefill (different shape class by design — vLLM never captures
    # it either) and run 2 is the first decode (warmup/autotune).
    if len(prev_runs) < 4 or seq.__dict__.get("_census_printed"):
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


def maybe_run(seq, skip_kills: bool, pre_op_callback) -> bool:
    """Replay fast path. True = this run was fully handled."""
    if os.environ.get("NBX_REPLAY_TUPLE_CENSUS") == "1":
        _install_seams()
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
    state = plans.get(sig)

    if isinstance(state, FrozenPlan):
        if os.environ.get("NBX_REPLAY_VERIFY_EVERY") == "1":
            # Diagnostic mode: verify EVERY replay, name the first
            # diverging occurrence per bucket.
            state.replay(seq._arena)
            replayed = _output_hashes(seq)
            seq._run_single_device(skip_kills, None)
            normal = _output_hashes(seq)
            n = seq.__dict__.setdefault("_replay_occurrence", {})
            n[sig] = n.get(sig, 0) + 1
            if replayed != normal:
                print(f"[Replay][VERIFY_EVERY] DIVERGENCE at "
                      f"occurrence {n[sig]} of bucket "
                      f"({state.launches} launches)")
            return True
        state.replay(seq._arena)
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
        seq._run_single_device(skip_kills, None)
        normal = _output_hashes(seq)
        if replayed == normal:
            plans[sig] = plan
            print(f"[Replay] plan VERIFIED byte-equal "
                  f"({plan.launches} launches) — replaying this bucket")
        else:
            plan.slab.retire()
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
                      tuple(_input_slot_range(seq)))
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
          f"{slab.size/1e6:.0f} MB, inputs {len(plan.frozen_inputs)}")
    return True
