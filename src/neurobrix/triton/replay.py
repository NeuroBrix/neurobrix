"""Frozen-plan replayer for the triton hot loop (Phase 4a: E1+E2+E6).

ONE-SHOT REQUESTS: recording is opt-in, off by default, and should stay off for
a single request — the plan does not outlive this process (only the slab SIZE
cache is written; there is no plan serialiser and no loader). Measured on an
M4 Pro, 120 tokens, byte-identical output: **+7.6% at a short prompt, 8.0x at
~480 tokens of context**. The cost is context-dependent; a single number
misstates it. Full record and the reasoning:
docs/reference/replay-recording-in-a-one-shot-request.md

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

from nbxtensor.tensor import NBXTensor
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.memory.slab import StepSlab
import logging
_log = logging.getLogger(__name__)

# THE ENGINE REPLAYS BY DEFAULT (the supervisor's order 2026-10-28 08:20, S3): a sequence that runs again with the same
# signature is measured, recorded, compared byte for byte against the normal path on one step, and only then replayed;
# every sequence it cannot vouch for is refused BY NAME once (`_decline`) and keeps the normal path. No environment switch:
# what decides is the sequence's own eligibility and that comparison. Measured on the rack (V100, 2026-10-28): TinyLlama-1.1B triton 120
# tokens at a ~600-word prompt 96.6 -> 17.2 ms a token, Qwen3-30B-A3B on 4 cards 383 -> 278 ms, texts byte-identical.
ENABLED = True

# THE COMPARISON IS A DIVERGENCE GUARD, NOT A PROOF (the supervisor's decision 2026-11-06 21:30). The verify step replays
# the plan, then runs the normal path on the same inputs and compares the graph outputs: it catches a plan that departs
# from the normal path (it refused whisper-large's noinv plan and Qwen3-30B-A3B's expert tail). It CANNOT see an error both
# passes share — a stale value both read, a wrong input both receive: on 2026-11-06 it matched 237 times on a wrong text.
# Its line therefore says what it observed and no more; no report, register or gate cites it as proof that an output is
# right — that proof is the entry point's run and its output looked at.
PLAN_MATCHED_LINE = ("[Replay] plan matches the normal path on this step (not a proof of the output) "
                     "({launches} launches) — replaying this bucket")

# A RECORDED PLAN DOES NOT OUTLIVE ITS PROCESS: plans live on the sequence object (`seq.__dict__["_replay_plans"]`),
# reused across the steps of one generation and discarded with the sequence (`retire_sequence_plans`). The only thing
# written to disk is the slab SIZE cache (`_store_slab_size`). The M4 Pro's 8.0x at ~480 tokens of context (2026-09-18)
# was measured before the paged KV decode replayed — every decode step then re-recorded; on the rack at a ~600-word
# prompt the replay is 5.6x FASTER (2026-10-28, above).


def _decline(seq, reason: str) -> None:
    """A sequence the replay cannot vouch for, said once per sequence and reason; it keeps the normal path."""
    said = seq.__dict__.setdefault("_replay_declined", set())
    if reason in said:
        return None
    said.add(reason)
    comp = str((getattr(seq, "dag", None) or {}).get("component_name", "?"))
    _log.debug(f"[Replay] {comp}: not replayed — {reason}; the normal path runs")
    return None

# Multi-device replay is LOCKED by default and this flag is how the lock is
# exercised for its equivalence proof — it is not a feature switch.
#
# The lock exists because a replay without the device action ran kernels on
# whatever card was current, and the slab served the wrong card's memory: both
# silently wrong rather than errors. Those two capabilities now exist
# (_SETDEV, per-device slabs), but "the capability exists" is not "the replay
# is correct". The lock comes off only when a multi-device replay is proven
# BIT-IDENTICAL to direct execution on a model that actually shards.
#
# Until that proof is recorded, this stays opt-in and the default path is
# unchanged. A lock that guards against a silent wrong answer is worth more
# than a capability opened early.
MULTIDEV_PROOF = os.environ.get("NBX_REPLAY_MULTIDEV") == "1"

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
from neurobrix.core.paths import engine_home as _engine_home
_SIZE_CACHE_DIR = os.path.join(str(_engine_home()), "replay_cache")


def _size_cache_path(seq) -> str:
    comp = str((seq.dag or {}).get("component_name", "component"))
    return os.path.join(_SIZE_CACHE_DIR, f"{comp}.json")


# the rule a stored slab size was computed by: a size from another rule is not this one's (2026-11-04: sizes measured as
# the live peak, then grown by a shortfall retry to 31.7 GB, survived every process and kept a 32 GB card full)
_SIZE_RULE = "first-fit-extent/1"


def _sig_hash(sig: tuple) -> str:
    import hashlib
    return hashlib.sha256(repr((_SIZE_RULE, sig)).encode()).hexdigest()[:16]


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

# The replay MECHANISM is the library's (`nbxtensor.backend.frozen_plan`, moved 2026-11-12: the owner's red line): the
# recording interceptor, the launch recorder and its seams, the frozen plan and its copies. What stays here is the
# engine's policy: when a sequence measures, records, verifies and replays, its signature, its plans' lifetimes.
from nbxtensor.backend import frozen_plan as FP  # noqa: E402
from nbxtensor.backend.frozen_plan import (  # noqa: E402
    FrozenPlan, STATE, H2D as _H2D, KERNEL as _KERNEL, MEMCPY as _MEMCPY, MEMSET as _MEMSET, SETDEV as _SETDEV,
    SYNC as _SYNC, arm_launch_recorder as _arm_launch_recorder, copy_into as _copy_into, drop_plan as _drop_plan,
    heap_copy as _heap_copy, install_seams as _install_seams, plan_of as _plan_of)
# where a recording broke: the first engine frame outside the replayer (the library names no application's files)
STATE.site_of = lambda filename: "neurobrix" in filename and "replay" not in filename

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
        _log.debug(f"[StepSlab] {comp}: stabilizing decode allocations "
                   f"({need / 1e6:.0f} MB slab)")
    slab = seq.__dict__.get("_stab_slab")
    if slab is not None:
        if not seq.__dict__.get("_stab_served_printed") and \
                seq.__dict__["_stab_run"] == 6:
            comp = str((seq.dag or {}).get("component_name", "?"))
            n_ops = len((seq.dag or {}).get("ops", {}))
            _log.debug(f"[StepSlab] {comp} (dag {n_ops} ops, seq id "
                       f"{id(seq) & 0xffff:04x}): served "
                       f"{len(slab.allocs) + len(slab.stale)} allocations "
                       f"last step (live {len(slab.allocs)}, "
                       f"stale {len(slab.stale)}, shortfall {slab.shortfall})")
            a, b = slab.prev_carve or [], slab.carve_sizes
            if a and b:
                if len(a) != len(b):
                    _log.debug(f"[StepSlab] {comp}: carve COUNT varies "
                               f"({len(a)} vs {len(b)})")
                else:
                    div = [i for i, (x, y) in enumerate(zip(a, b))
                           if x != y]
                    if div:
                        i = div[0]
                        _log.debug(f"[StepSlab] {comp}: carve sizes diverge "
                                   f"at {len(div)} indices; first at #{i}: "
                                   f"{a[max(0,i-2):i+3]} vs "
                                   f"{b[max(0,i-2):i+3]}")
                    else:
                        _log.debug(f"[StepSlab] {comp}: carve size sequence "
                                   f"IDENTICAL ({len(a)} allocs)")
            seq.__dict__["_stab_served_printed"] = True
        slab.reset()
    STATE.stab_slab = slab


# ---------------------------------------------------------------------------
# Seams — installed once; routed by the module-level state below
# ---------------------------------------------------------------------------



# Sequences that hold recorded plans (weak registry: the dict dies with
# the sequence, the SLABS do not — `retire_sequence_plans` is the release
# path an executor calls when it drops a sequence). The KV cache reaches
# the plans through `drop_plans_by_contribution` when it replaces a
# layer's buffers for a longer request — every plan recorded against the
# old addresses must go (D-SERVE-WARM-KV-GROWTH-ASYMMETRY: "replay-plan
# invalidation on growth"): a plan that survived the swap would replay
# kernel tuples, memcpys and captured graphs against freed or re-served
# memory. The signature LAYOUT is owned here (`signature()`): the owner
# contributions sit in one tuple at `sig[_CONTRIB_AT]`; callers match on a
# contribution, never on a position of the full key.
import weakref as _weakref
_PLAN_SEQS: "_weakref.WeakSet" = _weakref.WeakSet()


# segment). 33db72ee appended the segment after the flag and this read stayed at -2 — the flag, never a tuple — so a
# KV pool's growth retired none of its previous generation's plans (test_kv_prompt_aware_sizing, red since)
_CONTRIB_AT = -3


def _contributions(sig):
    if isinstance(sig, tuple) and len(sig) >= -_CONTRIB_AT and isinstance(sig[_CONTRIB_AT], tuple):
        return sig[_CONTRIB_AT]
    return ()


def drop_plans_by_contribution(pred) -> int:
    """Retire every recorded plan one of whose owner contributions
    satisfies `pred(contribution)` — frozen plans and pending VERIFY
    plans (slab retired, captured graph destroyed), plus the measuring /
    unreplayable states behind the same keys. Returns the number of keys
    dropped. Called by `PagedTritonKVCache` at a pool growth with a
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


def release_idle_slabs(dev: int) -> int:
    """A malloc refused on `dev` after the pool's flush (`DeviceAllocator.malloc_cuda`): every recorded plan holding a
    slab on `dev` between its runs is retired — a slab carries no live tensor between two runs of its sequence, it is
    cached memory exactly as the pool's free blocks are, and the plan Prism priced runs each component's activations,
    not a frozen copy of another's beside them. The sequence measures and records again at its next run. Returns the
    bytes released at once (a slab whose carved outputs are still alive frees when they die)."""
    freed = 0
    for seq in list(_PLAN_SEQS):
        plans = seq.__dict__.get("_replay_plans")
        if not plans:
            continue
        for sig in list(plans):
            plan = _plan_of(plans[sig])
            if plan is None or (plan.slab.dev_idx != dev and dev not in (getattr(plan, "other_slabs", None) or {})):
                continue
            slabs = [plan.slab] + list((getattr(plan, "other_slabs", None) or {}).values())
            now = sum(sl.size for sl in slabs if sl.dev_idx == dev and not sl.allocs)
            _drop_plan(plans, sig)
            freed += now
            comp = str((getattr(seq, "dag", None) or {}).get("component_name", "?"))
            _log.debug(f"[Replay] {comp}: a recorded plan's slab released on device {dev} ({now / 2**20:.0f} MB at once) — "
                       f"an allocation there needed the room; it records again at its next run")
    return freed


DeviceAllocator._idle_reclaimers.append(release_idle_slabs)


def retire_sequence_plans(seq) -> int:
    """Release every plan a sequence holds (its slabs and captured
    graphs) — the executor's teardown path when it drops the sequence.
    The weak registry alone does not free them: a retired-less slab
    stays in `nbxtensor.memory.slab.ACTIVE_SLABS` with its base allocation for the process."""
    plans = seq.__dict__.get("_replay_plans")
    if not plans:
        return 0
    n = 0
    for sig in list(plans):
        _drop_plan(plans, sig)
        n += 1
    _PLAN_SEQS.discard(seq)
    return n


# --- The NeuroBrix launcher's launches, recorded in the same shape ----------
#
# `_install_seams` records a step by wrapping Triton's `CompiledKernel.run`. On the
# Triton branch nothing calls that any more: the NeuroBrix launcher goes from its own
# binder straight to the driver (R33, third peel 2026-09-05). A step recorded through
# that seam alone holds the copies and NO kernel; the frozen plan then computes
# nothing and the verify gate rejects it — which is exactly what it did, costing the
# decode its CUDA-graph path without ever being wrong (measured 2026-09-08 on
# TinyLlama at a locked clock: 71.9 ms/token against 7.1 under Triton's launcher).
#
# So the launch is recorded in exactly the shape `_KERNEL` already carries, behind a
# small object that speaks `CompiledKernel`'s calling convention and dispatches to the
# engine's own driver. Every consumer — the direct replay, the graph capture, the
# tuple census, the error naming — therefore works unchanged, on either launcher.




def _input_slot_range(seq) -> range:
    """Arena layout is [weights | inputs | intermediates]."""
    return range(seq._num_weights, seq._num_weights + seq._num_inputs)


def _output_hashes(seq) -> Dict[str, str]:
    """sha256 of every graph-output tensor's LOGICAL values (its shape, then its elements in order; D2H once — verify
    runs once per bucket per process). Hashed from the storage, a view's bytes matched a scrambled copy of them, and a
    plan whose permuted output came back in the wrong order matched (Allegro-TI2V's encoder, 2026-11-04)."""
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
        t = t.contiguous()
        buf = (ctypes.c_char * t._nbytes)()
        DeviceAllocator.memcpy(ctypes.addressof(buf), t.data_ptr(),
                               t._nbytes, 2)
        out[tid] = hashlib.sha256(repr(tuple(t.shape)).encode() + bytes(buf)).hexdigest()
    return out


def signature(seq) -> Optional[tuple]:
    # A tail replayed after a head that runs directly (`TritonSequence._run_expert_tail`): the sequence is multi-device
    # for its head's host experts only, the tail's ops run on their own cards and its cross-card copies are recorded.
    if (getattr(seq, "_is_multi_device", False) and not MULTIDEV_PROOF
            and not getattr(seq, "_tail_replay", False)):
        return _decline(seq, "its ops run on several cards and a multi-device replay is not proven bit-identical yet "
                             "(only an expert store's card-held tail is)")
    # Stage-driven sequences (the VibeVoice tokenizers the next_token_diffusion
    # flow marks — formerly the core/flow/stages/ calling convention) live outside the standard
    # flow contract: five falsified byte-divergence mechanisms on the
    # VibeVoice tokenizers (2026-08-13 adjudication log) put them
    # behind the capability gate until the residual is root-caused.
    if getattr(seq, "_replay_ineligible", False):
        return _decline(seq, "a stage-driven sequence (outside the standard flow contract)")
    # A run that frees inputs its caller hands over (a layer-streamed piece's `_consumed_input_tids`) frees their memory
    # at their last use: a frozen plan can neither keep the buffer its launches read nor refill it. granite-speech-3.3-8b
    # triton on the M4 Pro: replayed, its pieces read a freed input (bmm::0 None), then, kept, a stale one (2026-11-06).
    if getattr(seq, "_consumed_input_tids", None):
        return _decline(seq, "it frees the inputs its caller hands over at their last use (a layer-streamed piece)")
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
            if owner is None or not all(hasattr(owner, m) for m in ("replay_signature", "replay_advance",
                                                                  "replay_restore", "replay_host_mark")):
                return _decline(seq, f"interceptor {getattr(func, '__qualname__', func)!r} keeps a state it does not "
                                     f"register with the replay (replay_signature/advance/restore/host_mark)")
            by_owner.setdefault(id(owner), (owner, []))[1].append(func)
        for owner, funcs in by_owner.values():
            s = owner.replay_signature(funcs)
            if s is None:
                return _decline(seq, f"{type(owner).__name__} gives no replay signature for this step (a prefill, or a "
                                     f"position scalar its launches bake)")
            contrib.append(s)
            owners.append(owner)
    seq.__dict__["_replay_state_owners"] = owners
    # a kept invariant state (state engine step 4) enters the bucket by its addresses; a run evaluating it is not replayed
    inv = getattr(seq, "_invariant", None)
    if inv is not None:
        ident = inv.replay_identity(seq._arena)
        if ident is None:
            return _decline(seq, "its invariant state is evaluated at this run (kept for the next)")
        contrib.append(("invariant",) + ident)
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
        return _decline(seq, "an in-graph random op draws from host-advanced generator state a recorded launch freezes")
    parts: List[Any] = []
    for s in _input_slot_range(seq):
        t = seq._arena[s]
        if t is None:
            parts.append((s, None))
        else:
            parts.append((s, tuple(t.shape), int(t.nbx_dtype)))
    # Every symbol's bound value: a launch bakes the extents its symbols resolved to, so two runs whose bindings differ
    # are two plans. The Triton resolver's are `bindings` (`triton/symbols.SymbolResolver`); this line read a
    # `resolved` it never had, so no symbol ever entered the signature — deepseek-moe-16b-chat's rotary table,
    # `arange(nbx_rope_len + 1)` with `nbx_rope_len` bound per step from position_ids, was recorded at one step's length
    # and replayed at the next step's position: index_select out of range, CUresult 710 (gate 2026-11-01, bisected to
    # 18fb11c2, the first commit that replayed that component's decode step).
    resolver = getattr(seq, "_symbol_resolver", None)
    bindings = getattr(resolver, "bindings", None) if resolver is not None else None
    if isinstance(bindings, dict):
        parts.append(tuple(sorted(
            (k, v) for k, v in bindings.items()
            if isinstance(v, (int, float, str)))))
    parts.append(tuple(sorted(seq._op_uid_interceptors)))
    parts.append(tuple(sorted(seq._op_interceptors)))
    # Registration-contract state (e.g. the KV decode bucket): a state
    # change — bucket boundary crossed — lands the run in a NEW bucket
    # (measure/record/verify again) instead of replaying stale extents.
    parts.append(tuple(contrib))
    parts.append(bool(getattr(seq, "_activations_fp16_safe", False)))
    # a decode step replayed in segments (`TritonSequence._expert_segments`): one plan per segment
    parts.append(getattr(seq, "_replay_segment", None))
    sig = tuple(parts)
    assert sig[_CONTRIB_AT] is parts[-3], "the contributions' place (_CONTRIB_AT) and the signature's layout disagree"
    if os.environ.get("NBX_REPLAY_SIG_DIAG") == "1":
        prev = seq.__dict__.get("_sig_diag_prev")
        if prev is not None and prev != sig and len(prev) == len(sig):
            for i, (a, b) in enumerate(zip(prev, sig)):
                if a != b:
                    _log.debug(f"[Replay][SIG] part {i} changed: "
                               f"{repr(a)[:200]} -> {repr(b)[:200]}")
            for o in owners:
                _log.debug(f"[Replay][SIG] owner id={id(o) & 0xffffff:06x} "
                           f"cache_len={o.get_cache_len()}")
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
        from nbxtensor.ops.autotune_cache import seed as _at_seed
        n = _at_seed()
        if n:
            _log.debug(f"[Replay] autotune artifact seeded: {n} configs")
    except (OSError, ValueError):
        pass  # artifact I/O is an optimization, never a failure source
    except Exception as e:  # API drift must stay observable, once
        _log.debug(f"[Replay] autotune seed unavailable ({type(e).__name__}: {e})")


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
    _arm_launch_recorder(True)
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
        _log.debug(f"[TupleCensus] {comp}: launch COUNT varies "
                   f"({len(a)} vs {len(b)}) — structural per-step change")
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
    _log.debug(f"[TupleCensus] {comp}: {len(a)} launches, "
               f"{len(varying)} vary between consecutive runs")
    from collections import Counter
    by_kernel = Counter((v[1], v[2], tuple(v[3])) for v in varying)
    for (kname, gdiff, args), cnt in by_kernel.most_common(12):
        _log.debug(f"  {cnt:5d}x {kname}  grid_varies={gdiff}  "
                   f"varying_arg_positions={list(args)}")
    seq.__dict__["_census_printed"] = True


def _mem_diag(tag: str) -> None:
    """NBX_REPLAY_MEM_DIAG=1: print the allocator's live-tracked bytes
    at replay state transitions (retention forensics)."""
    if os.environ.get("NBX_REPLAY_MEM_DIAG") != "1":
        return
    total = sum(DeviceAllocator._cuda_live_bytes.values())
    _log.debug(f"[Replay][MEM] {tag}: live_tracked={total/1e6:.0f}MB")


def _advance_what_the_recording_moved(plan, owners) -> None:
    """After a replayed run: mirror onto each state owner's host state what the recorded NORMAL run moved there, and
    nothing else (`replay_advance(before, after)`, the owner's marks around the recorded run). A replay that mirrored a
    fixed "one step" was right only for a sequence holding the whole step. A layer-streamed model replays the one piece
    that frees no input its caller hands over — its first piece, with the first layer's attention in it: its replay
    advanced the cache's length although no last layer ran, and did NOT count the attention call it made, so every
    later piece's attention took the layer number of the one before it (keys and values written into the wrong layer)
    and the last layer's never came (TinyLlama, Triton, a 1 600 MB card: "The capital of Francephстутяш86488ш",
    2026-11-11; at 2 000 MB no piece replays and the text is right)."""
    for i, o in enumerate(owners):
        moved = plan.advances.get(i)
        if moved is not None:
            o.replay_advance(*moved)


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



def _outputs_off_the_slab(seq, slab) -> int:
    """Every graph output of `seq` carved from `slab` replaced in the arena by a heap copy; the count. A retired slab is
    freed only once nothing carved from it lives, and the caller keeps a run's outputs."""
    lo, hi, n = slab.base, slab.base + slab.size, 0
    for tid in (seq.dag.get("output_tensor_ids") or []):
        slot = seq._tid_to_slot.get(tid)
        t = seq._arena[slot] if slot is not None else None
        if t is None or not (lo <= t.data_ptr() < hi):
            continue
        seq._arena[slot] = _heap_copy(t)
        n += 1
    return n


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
    if not ENABLED:
        return False
    if pre_op_callback is not None:
        _decline(seq, "a per-op callback runs beside its ops (a zero3 ratchet, a calibration), which a frozen launch "
                      "plan cannot call")
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
                _log.debug(f"[Replay][VERIFY_EVERY] DIVERGENCE at "
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
                        fro, nor = FP.head_values(frozen), FP.head_values(seq._arena[slot])
                        _log.debug(f"[Replay][VERIFY_EVERY] out[:8] "
                                   f"replayed(frozen)={fro[:8].tolist()} "
                                   f"normal={nor[:8].tolist()}")
            return True
        state.replay(seq._arena)
        _advance_what_the_recording_moved(state, owners if getattr(seq, "_replay_advances", True) else ())   # a head segment: none
        n_r = seq.__dict__.setdefault("_replay_count", {})
        n_r[sig] = n_r.get(sig, 0) + 1
        if n_r[sig] % 50 == 1:
            _mem_diag(f"replay #{n_r[sig]}")
        if os.environ.get("NBX_REPLAY_GRAPH_DUMP") == "1" and owners:
            for o in owners:
                got = o.cache.replay_position_diag()
                if got is None:
                    continue
                _log.debug(f"[Replay][GRAPH] post-replay dev_pos={got[0]} "
                           f"host_len={got[1]}")
        return True
    if isinstance(state, tuple) and state[0] == "VERIFY":
        # verify-first-replay (a divergence guard, battery adjudication
        # 2026-08-13: a vibevoice stage-handler bucket carried
        # per-step state none of the static guards see). Replay, hash
        # the graph outputs, then run the NORMAL path on the same
        # inputs and compare: equal -> the plan is kept (it matches the
        # normal path on this step, nothing more: an error common to
        # both passes is invisible here, see PLAN_MATCHED_LINE);
        # different -> plan discarded loudly. Either way this step's
        # results come from the NORMAL path. One doubled step per
        # bucket per process.
        plan = state[1]
        # a tail's boundary slots (the head's outputs it reads) die inside the tail: kept here, put back for the normal
        # re-run below, which reads them as the replay did
        _boundary = {sl: seq._arena[sl] for sl in plan.extra_scan}
        _tail = bool(getattr(seq, "_tail_replay", False))
        if _tail and any(not hasattr(o, "replay_lens_snapshot")
                         or not hasattr(getattr(o, "cache", o), "replay_lens_snapshot") for o in owners):
            plans[sig] = "UNREPLAYABLE"
            _log.debug("[Replay] tail declined: a state owner cannot rewind exactly what the tail advanced")
            seq._run_single_device(skip_kills, None)
            return True
        _snaps = [(o, o.replay_lens_snapshot()) for o in owners] if _tail else []
        plan.replay(seq._arena)
        replayed = _output_hashes(seq)
        # Registration contract: rewind the replayed pass's
        # non-idempotent device writes (e.g. KV position counters) so
        # the normal re-run below operates on the SAME state — it then
        # advances host+device consistently itself.
        if _tail:
            for o, sn in _snaps:
                o.replay_lens_restore(sn)
        else:
            for o in owners:
                o.replay_restore()
        for sl, t in _boundary.items():
            seq._arena[sl] = t
        seq._run_single_device(skip_kills, None)
        normal = _output_hashes(seq)
        if replayed == normal:
            plans[sig] = plan
            _log.debug("%s", PLAN_MATCHED_LINE.format(launches=plan.launches))
            _mem_diag("post-verify")
        else:
            FP.retire(plan)
            plans[sig] = "UNREPLAYABLE"
            _log.debug("[Replay] plan REJECTED at verify (outputs differ "
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
        # MEASURE pass: normal run under the watermark counter (the library's `measure`).
        own, other_cards, peak = FP.measure(lambda: seq._run_single_device(skip_kills, None), seq.device_idx)
        plans[sig] = ("MEASURED", own, 1)
        seq.__dict__.setdefault("_replay_live_peak", {})[sig] = peak   # what the verify pass holds
        # every OTHER card the run allocated on gets its own slab at recording (a held expert's card, S3): served from
        # the sequence's card's slab, its tensors had addresses on the wrong card (an illegal access at replay)
        seq.__dict__.setdefault("_replay_dev_hw", {})[sig] = other_cards
        _mem_diag("post-measure")
        return True

    # ("MEASURED", slab_bytes, attempt) — record this execution.
    _, slab_bytes, attempt = state
    if slab_bytes <= 0:
        plans[sig] = "UNREPLAYABLE"
        return False
    # The slab is the recording's whole memory: the blocks the measured run left in the pool (and the expandable
    # range's free chunks) go back to the driver first, and the recording is made only where its slab AND the verify
    # pass that runs the normal path beside it fit what the card has free — a plan that cannot is the normal path's,
    # never an out-of-memory (Allegro-TI2V's encoder tile, 2026-11-04: the measured run's 9.9 GB stayed cached under a
    # 12.7 GB slab, and the verify pass beside them, with the pipeline's resident weights, overran a 32 GB card).
    _dev = seq.device_idx if seq.device_idx is not None else DeviceAllocator.get_device()
    _verify = int((seq.__dict__.get("_replay_live_peak") or {}).get(sig, slab_bytes))
    _fits, _free = FP.room_for(_dev, slab_bytes, _verify)
    if not _fits:
        comp = str((getattr(seq, "dag", None) or {}).get("component_name", "?"))
        _log.debug(f"[Replay] {comp}: not recorded — its slab ({slab_bytes / 2**20:.0f} MB) and the verify pass beside it "
                   f"({_verify / 2**20:.0f} MB) need more than the {_free / 2**20:.0f} MB free on device {_dev}; the normal "
                   f"path keeps this graph")
        plans[sig] = "UNREPLAYABLE"
        return False
    try:
        slab, others = FP.slabs(slab_bytes, seq.device_idx, seq.__dict__.get("_replay_dev_hw", {}).get(sig) or {})
    except Exception as e:
        _log.debug(f"[Replay] slab alloc failed ({slab_bytes/1e6:.0f} MB, or on another card): "
                   f"{e} — bucket keeps the normal path")
        plans[sig] = "UNREPLAYABLE"
        return False
    _marks = [o.replay_host_mark() for o in owners]
    records, failed, broken = FP.record(lambda: seq._run_single_device(skip_kills, None), slab, others)

    if failed is not None or broken is not None:
        reason = broken or f"exception: {failed}"
        shortfall = slab.shortfall
        # The run's outputs were carved from the slab and the caller keeps them: handed over as heap copies (as a
        # frozen plan's are), so the retired slab is freed once the run's own tensors die, not held beside the rest of
        # the run. Kept, a VAE chunk's output pinned a 7.5 GB undersized slab through the next chunks (Allegro triton
        # on a V100-16GB, 2026-11-01: the 2.5 GB malloc that followed was refused).
        if failed is None:
            _outputs_off_the_slab(seq, slab)
        slab.retire()  # deferred: frees only after carved tensors die
        for _o in others.values():
            _o.retire()
        if failed is None and shortfall > 0 and attempt < 2:
            # Data-driven retry: the failed attempt measured the exact
            # extra bytes needed.
            plans[sig] = ("MEASURED", slab_bytes + shortfall, attempt + 1)
            _log.debug(f"[Replay] slab undersized by {shortfall/1e6:.0f} MB — "
                       f"will re-record at {(slab_bytes+shortfall)/1e6:.0f} MB")
            return True  # the recording run still executed correctly
        plans[sig] = "UNREPLAYABLE"
        _log.debug(f"[Replay] bucket UNREPLAYABLE ({reason}) — normal path "
                   f"keeps this graph")
        if failed is not None:
            # The failed partial run may have left the arena mid-state:
            # re-execute normally from the top (ops are deterministic
            # and slot writes are idempotent across a rerun).
            seq._run_single_device(skip_kills, None)
        return True

    _skip, _lim = getattr(seq, "_run_skip", None), seq._num_weights + seq._num_inputs
    _read = None
    if _skip is not None and getattr(seq, "_tail_replay", False):
        _read = tuple(sorted({sl for i, op in enumerate(seq._ops) if not _skip[i]
                              for sl in (op.all_input_slots or ()) if sl < _lim}))
    plan = FrozenPlan(records, slab, list(seq._arena),
                      tuple(_input_slot_range(seq)),
                      scan_limit=_lim,
                      extra_scan=tuple(getattr(seq, "_replay_extra_scan", ()) or ()),
                      scan_slots=_read)
    plan.other_slabs = others
    plan.advances = {i: (_marks[i], o.replay_host_mark()) for i, o in enumerate(owners)}
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
    plan.hand_out_outputs(seq._arena)
    _store_slab_size(seq, sig, slab.size)
    # E2-full: the recording pass just warmed the sanctioned autotuners
    # for every shape in this graph — persist the selected configs into
    # the arch-keyed artifact (seeded back on the next process's
    # install; immune to triton's source-hash invalidation).
    try:
        from nbxtensor.ops.autotune_cache import capture as _at_capture
        _at_capture()
    except (OSError, ValueError):
        pass  # artifact I/O is an optimization, never a failure source
    except Exception as e:  # API drift must stay observable, once
        _log.debug(f"[Replay] autotune capture unavailable "
                   f"({type(e).__name__}: {e})")
    comp = str((seq.dag or {}).get("component_name", "?"))
    _log.debug(f"[Replay] plan frozen[{comp}]: {plan.launches} launches "
               f"({plan.actions} actions), slab "
               f"{slab.size/1e6:.0f} MB, inputs {len(plan.frozen_inputs)}, "
               f"outputs {len(plan.output_slots)}")
    _mem_diag("post-record")
    return True
