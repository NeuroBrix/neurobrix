"""The frozen plan: a device step recorded once and replayed as direct launches (nbxtensor's replay mechanism).

Moved from the NeuroBrix engine (`triton/replay.py`, 2026-11-12: the owner's red line — replay is tensor work and lives
in the library). What is here is MECHANISM: the recording interceptor (`STATE`: the allocator's copies, sets and
synchronisations, the launcher's launches, recorded as actions), the seams that arm it, the plan that replays the
actions (directly, or as captured device graphs: `replay_graph`) and the copies a replay makes into its frozen
buffers. WHEN a step is recorded or replayed — its signature, its eligibility, the verify pass — is the caller's.

The actions: `KERNEL` (kernel, g0, g1, g2, stream, flat values), `MEMCPY` (dst, src, nbytes, kind), `MEMSET` (ptr,
value, nbytes), `SETDEV` (device), `SYNC` (device), `H2D` (dst, host bytes snapshot).
"""
from __future__ import annotations

import logging
import os
import traceback
from typing import Dict, List, Optional, Tuple

from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.memory.slab import SlabAllocator, SlabInterceptor
from nbxtensor.tensor import NBXTensor

_log = logging.getLogger(__name__)

KERNEL = "k"     # (kernel, g0, g1, g2, stream, flat_vals)
MEMCPY = "m"     # (dst, src, nbytes, kind) — device-side pointers only
MEMSET = "s"     # (ptr, value, nbytes)
SETDEV = "d"     # (device_idx,) — the card the following actions run on.
                  # Emitted only when the device CHANGES, so a single-device
                  # plan carries one of these and costs nothing. Without it a
                  # replay runs every recorded launch on whatever card happens
                  # to be current: silently wrong, not an error. One of the two
                  # reasons multi-device was locked out of replay.
SYNC = "y"       # (device_idx,) — a device synchronisation the recorded run made (`transfer_tensor` waits for the
                  # source card before a peer copy: the copy is not ordered after another card's kernels). Replayed
                  # without it, a held expert's card read a buffer still being written: error 700 that
                  # CUDA_LAUNCH_BLOCKING hid (Qwen3-30B-A3B tail replay, 2026-10-28).
H2D = "h"        # (dst, host_snapshot: bytes) — the recorded host
                  # source is transient (probe 5: a dangling 16-B host
                  # staging pointer at replay); the plan owns a byte
                  # snapshot instead. A step-VARYING H2D would replay
                  # stale bytes — the byte gates adjudicate that class
                  # (contract: byte mismatch = replay bug/refusal).



class _State(SlabInterceptor):
    """The replayer's state, and the allocator's interceptor while it is installed (`install_seams`): the slab
    routing and the watermark measurement are the library's (`nbxtensor.memory.slab.SlabInterceptor`); what is the
    replayer's own is the recording — the copies, sets and synchronisations a plan must reproduce."""

    def __init__(self) -> None:
        super().__init__()
        self.records: List[Tuple[str, tuple]] = []
        self.broken: Optional[str] = None
        self.rec_dev: Optional[int] = None
        # P-REPLAY-KV-DECODE step A: tuple census (pure observation —
        # per-run launch-tuple capture + consecutive-run diff; no
        # eligibility gates, no replay ever). NBX_REPLAY_TUPLE_CENSUS=1.
        self.census_current: Optional[List[tuple]] = None
        # Census attribution: the seq whose run the current capture
        # belongs to (armed at its tick; banked at the NEXT tick).
        self.census_owner = None
        # Measurement attribution (same off-by-one shape): the seq
        # whose run the measuring window covers.
        self.measure_owner = None
        # which stack frame names where a recording broke: a predicate on its file name, the application's (it knows
        # its own files); none set, the reason carries no site
        self.site_of = lambda filename: False

    def open_measurement(self) -> None:
        """A fresh watermark window: every counter of the measured run at zero."""
        self.measuring = True
        self.live = 0
        self.high_water = 0
        self.measured = {}
        self.live_by_dev, self.high_water_by_dev = {}, {}
        self.events_by_dev, self._key_of_ptr = {}, {}

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
            self.records.append((SETDEV, (int(dev),)))

    def break_plan(self, reason: str) -> None:
        if self.broken is None:
            site = ""
            for fr in reversed(traceback.extract_stack()[:-2]):
                if self.site_of(fr.filename):
                    site = f" at {os.path.basename(fr.filename)}:{fr.lineno}"
                    break
            self.broken = reason + site

    def slab_refused(self, reason: str) -> None:
        self.break_plan(reason)

    def memcpy(self, direct, dst, src, nbytes, kind):
        if self.recording:
            if kind == 2:
                self.break_plan("D2H memcpy during recording")
            elif kind == 1:
                import ctypes
                # bytes of any size: `ctypes.string_at` takes a C int and refuses 2 GiB
                snap = bytes((ctypes.c_uint8 * int(nbytes)).from_address(int(src))) if int(nbytes) else b""
                self.note_device()
                self.records.append((H2D, (int(dst), snap)))
            else:
                self.note_device()
                self.records.append((MEMCPY, (int(dst), int(src),
                                               int(nbytes), kind)))
        return direct(dst, src, nbytes, kind)

    def memset(self, direct, ptr, value, nbytes):
        if self.recording:
            self.note_device()
            self.records.append((MEMSET, (int(ptr), int(value),
                                           int(nbytes))))
        return direct(ptr, value, nbytes)

    def sync(self, direct):
        if self.recording:
            self.records.append((SYNC, (int(DeviceAllocator.get_device()),)))
        return direct()


STATE = _State()



def plan_of(state):
    """The FrozenPlan behind a plans-dict state, if it holds resources:
    a frozen plan, or a pending ("VERIFY", plan) whose slab was already
    allocated and whose verify pass has not run yet."""
    if isinstance(state, FrozenPlan):
        return state
    if isinstance(state, tuple) and len(state) == 2 and state[0] == "VERIFY" \
            and isinstance(state[1], FrozenPlan):
        return state[1]
    return None


def drop_plan(plans: dict, sig) -> None:
    plan = plan_of(plans.pop(sig, None))
    if plan is not None:
        plan.slab.retire()   # deferred: frees once carved tensors die
        for _o in (getattr(plan, "other_slabs", None) or {}).values():
            _o.retire()      # the other cards' slabs (a held expert's card) go with it
        if plan.graph is not None:
            try:
                plan.graph.destroy()
            finally:
                plan.graph = None


# where `signature` places the registered owners' contributions: two parts follow them (the fp16-safe flag, the



_INSTALLED = False


class _NBXLaunch:
    """One prepared kernel, called the way `CompiledKernel.run` is called."""

    __slots__ = ("prep", "kinds", "name", "function", "packed_metadata", "names", "types", "extra")

    def __init__(self, prep, kinds, extents=None):
        self.prep = prep
        self.kinds = kinds
        # the pointers' extents for a by-extent driver: dropped, Metal wrapped each to the slab's end (RUN 40, 20.9 GB)
        self.extra = {} if extents is None else {"extents": list(extents)}
        self.name = prep.name
        self.function = prep.function
        self.packed_metadata = None
        # The launcher hands a driver the parameter NAMES and TYPES beside the
        # values, and a backend that packs its scalars into one buffer computes
        # the field offsets FROM THOSE TYPES. The recorded tuple carries only
        # `(kind, value)`, so replay used to drop both — which on triton-ext is
        # a refusal by name ("Launching without them would pack to the wrong
        # offsets WITHOUT failing"), and the replay path simply could not drive
        # that backend. The fork's driver did not need them, so nothing said so.
        #
        # They do not belong in the RECORD: they are a property of the
        # compilation, identical for every launch of this specialisation, and
        # `prep.signature` already holds them. Derived here, once per adapter.
        runtime = [(n, t) for n, t in prep.signature.items() if t != "constexpr"]
        self.names = [n for n, _t in runtime]
        self.types = [t for _n, t in runtime]

    def run(self, g0, g1, g2, stream, function, packed_metadata,
            launch_md, enter_hook, exit_hook, *vals):
        # The stream is the CALLER's: the graph capture hands its own capture stream
        # because the recorded one may be the uncapturable legacy stream.
        from nbxtensor.backend.launcher import active_driver
        # `trailing` travels with the PREPARATION, not with the recorded tuple:
        # it is what the compiled kernel declares beyond the call, so dropping
        # it here would replay a launch that binds fewer buffers than the kernel
        # has — the silent class this seam exists to end.
        active_driver().launch(self.prep.function, (g0, g1, g2), self.prep.block,
                               self.prep.shared, stream,
                               list(zip(self.kinds, vals)),
                               names=self.names, types=self.types,
                               trailing=self.prep.trailing, **self.extra)


_NBX_ADAPTERS: Dict[tuple, "_NBXLaunch"] = {}


def _nbx_record(prep, grid, params, extents=None) -> None:
    """The launcher's recorder seam (`launcher.set_launch_recorder`), armed for the Triton property's window."""
    flat = tuple(v for _kind, v in params)
    if STATE.census_current is not None and not STATE.recording:
        STATE.census_current.append((prep.name, grid[0], grid[1], grid[2], flat))
        return
    if not STATE.recording:
        return
    key = (id(prep), None if extents is None else tuple(extents))
    adapter = _NBX_ADAPTERS.get(key)
    if adapter is None or len(adapter.kinds) != len(params):
        adapter = _NBX_ADAPTERS[key] = _NBXLaunch(prep, tuple(k for k, _v in params), key[1])
    STATE.note_device()
    # The recorded stream is never read back (the replay and the capture each supply
    # their own), and this launcher's is the legacy one: it is stored as 0 for shape.
    STATE.records.append((KERNEL, (adapter, grid[0], grid[1], grid[2], 0, flat)))


def arm_launch_recorder(on: bool) -> None:
    from nbxtensor.backend import launcher as _lnch
    _lnch.set_launch_recorder(_nbx_record if on else None)


def _recording_run_property(orig_run_prop):
    """Triton's `CompiledKernel.run` property, wrapped: under the census a launch is noted, under a recording it is
    recorded as a KERNEL action; otherwise the kernel's own launcher, untouched."""
    def _run_prop(self):  # noqa: ANN001
        raw = orig_run_prop.fget(self)
        if STATE.census_current is not None and not STATE.recording:
            return _census_run(self, raw)
        if not STATE.recording:
            return raw
        return _recording_run(self, raw)
    return property(_run_prop)


def _census_run(kernel, raw):
    def census_run(g0, g1, g2, stream, function, packed_metadata,
                   launch_md, enter_hook, exit_hook, *vals):
        flat = tuple(
            int(v.data_ptr()) if hasattr(v, "data_ptr") else v
            for v in vals)
        STATE.census_current.append(
            (getattr(kernel, "name", "?"), g0, g1, g2, flat))
        return raw(g0, g1, g2, stream, function, packed_metadata,
                   launch_md, enter_hook, exit_hook, *vals)
    return census_run


def _recording_run(kernel, raw):
    def recording_run(g0, g1, g2, stream, function, packed_metadata,
                      launch_md, enter_hook, exit_hook, *vals):
        flat = tuple(
            int(v.data_ptr()) if hasattr(v, "data_ptr") else v
            for v in vals)
        STATE.note_device()
        STATE.records.append(
            (KERNEL, (kernel, g0, g1, g2, stream, flat)))
        return raw(g0, g1, g2, stream, function, packed_metadata,
                   launch_md, enter_hook, exit_hook, *vals)
    return recording_run


def install_seams() -> None:
    """Install the recording seams once: Triton's launch property, the allocator's interceptor (`STATE`), and the
    host read that breaks a recording (`NBXTensor.item`)."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from triton.compiler.compiler import CompiledKernel

    CompiledKernel.run = _recording_run_property(CompiledKernel.run)

    DeviceAllocator.install_interceptor(STATE)

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
                 "graph", "graph_disabled", "extra_scan", "other_slabs", "_crosses", "scan_slots", "advances")

    def __init__(self, records: List[Tuple[str, tuple]],
                 slab: SlabAllocator,
                 arena_list: List[Optional[NBXTensor]],
                 input_slots: Tuple[int, ...],
                 scan_limit: Optional[int] = None,
                 extra_scan: Tuple[int, ...] = (),
                 scan_slots: Optional[Tuple[int, ...]] = None) -> None:
        self.records = records
        # {owner index in the sequence's `_replay_state_owners`: (its host mark before, after) the recorded run}: a
        # replay mirrors exactly that move (`_advance_what_the_recording_moved`)
        self.advances: Dict[int, tuple] = {}
        # the weight/input slots a SEGMENT's ops read (a decode step replayed in segments): its scan visits only those —
        # nine segments scanning every weight slot cost more a token than they saved (2026-10-28). None: the whole
        # bound region, as a whole-run plan always did.
        self.scan_slots = scan_slots
        self.slab = slab  # strong ref pins the slab for the plan's life
        self.arena_snapshot = list(arena_list)
        self.frozen_inputs: Dict[int, NBXTensor] = {}
        for s in input_slots:
            t = arena_list[s]
            if t is not None:
                self.frozen_inputs[s] = t
        self.launches = sum(1 for r in records if r[0] == KERNEL)
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
        # Intermediate slots a plan READS that it does not produce: a tail recorded after a head that runs directly
        # (the caller's) reads the head's outputs, which land at new addresses every step — copied
        # into the frozen buffers like an input.
        self.extra_scan: Tuple[int, ...] = tuple(s for s in extra_scan if s >= self.scan_limit)
        # the slabs of the OTHER cards the plan allocated on (held experts' cards): pinned for the plan's life
        self.other_slabs: Dict[int, "SlabAllocator"] = {}
        # Phase 4b: lazily captured CUDA graph of the action list
        # (opt-in NBX_REPLAY_GRAPH=1, A/B-gated adoption). A capture
        # failure disables the graph for this bucket only — direct
        # replay stays the correct path.
        self.graph = None
        self.graph_disabled = False

    def replay(self, arena) -> None:
        """Replay the plan over `arena` (an `nbxtensor.memory.arena.Arena`): the rebound inputs copied into the frozen
        buffers, the actions run (as captured graphs or directly), the recorded arena view restored, the outputs
        handed out as fresh copies."""
        # Freeze-and-copy over rebound WEIGHT/CONSTANT/INPUT slots:
        # anything rebound between steps with a NEW object (bind_inputs
        # slots, but also per-step re-sliced seq-dependent constants —
        # probe 4 caught a strided_copy at replay action 6 reading a
        # freed previous-step constant) gets its bytes copied into the
        # frozen buffer so every recorded pointer stays live and
        # current. Identity scan cost is ~µs; identical objects skip.
        _scan_copied = self._scan_bound(arena) + self._scan_extra(arena)
        if os.environ.get("NBX_REPLAY_GRAPH_DUMP") == "1":
            self._dump_small_inputs(_scan_copied)
        # A plan that crosses cards (an expert store's card-held tail) replays one graph per stretch on one card
        # (`replay_graph.SegmentedPlan`), the engine's choice: its direct replay is ~1 850 Python-driven launches a
        # token (Qwen3-30B-A3B on 4 V100, 2026-10-28: ~60 ms of a 184 ms token for a few ms of kernels).
        _segmented = self.crosses_cards() or self.scan_slots is not None     # a segment of a step: graphs too
        if ((os.environ.get("NBX_REPLAY_GRAPH") == "1" or _segmented)
                and not self.graph_disabled):
            self._replay_graph(_segmented)
        else:
            self._direct_actions()
        arena.restore_from(self.arena_snapshot)
        self.hand_out_outputs(arena)

    def _scan_bound(self, arena) -> int:
        """The weight/input slots (or a segment's own) rebound since the recording: their values copied into the
        frozen buffers; the count."""
        _scan_copied = 0
        # C-level pairing (zip + islice over the raw slot list) — the
        # indexed per-slot form cost ~19k Arena.__getitem__ per decode
        # token on the 30B row (2026-08-23 host profile).
        from itertools import islice
        if self.scan_slots is not None:
            _snap, _cur = self.arena_snapshot, arena.slots
            pairs = ((sl, (_snap[sl], _cur[sl])) for sl in self.scan_slots)
        else:
            pairs = enumerate(islice(zip(self.arena_snapshot, arena.slots), self.scan_limit))
        for slot, (frozen, new) in pairs:
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
            _copy_slot(slot, frozen, new)
        return _scan_copied

    def _scan_extra(self, arena) -> int:
        """The intermediate slots the plan reads without producing them (`extra_scan`), copied in; the count."""
        _scan_copied = 0
        for slot in self.extra_scan:
            frozen, new = self.arena_snapshot[slot], arena.slots[slot]
            if frozen is None or new is None or new is frozen:
                continue
            if (tuple(new.shape) != tuple(frozen.shape)
                    or new.nbx_dtype != frozen.nbx_dtype):
                raise RuntimeError(
                    f"ZERO FALLBACK: replay boundary slot {slot}: {tuple(new.shape)}/{new.nbx_dtype} vs frozen "
                    f"{tuple(frozen.shape)}/{frozen.nbx_dtype} — the bucket signature should have caught this")
            _scan_copied += 1
            DeviceAllocator.set_device(frozen.device_idx)
            copy_into(frozen, new)
        return _scan_copied

    def _dump_small_inputs(self, _scan_copied: int) -> None:
        import ctypes as _cs
        vals = []
        for s, ft in self.frozen_inputs.items():
            if ft.nbytes() <= 16:
                b = (_cs.c_int64 * (ft.nbytes() // 8))()
                DeviceAllocator.memcpy(_cs.addressof(b),
                                       ft.data_ptr(), ft.nbytes(), 2)
                vals.append((s, list(b)))
        _log.debug(f"[Replay][SCAN] copied={_scan_copied} "
                   f"small_inputs={vals}")

    def _replay_graph(self, _segmented: bool) -> None:
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
                self._capture(recs, cut, _segmented)
            self.graph.launch()
        except Exception as e:
            self._graph_refused(e)
        else:
            # Bisection tail (order-preserving): runs OUTSIDE the
            # refusal try — a tail failure is a direct-replay
            # failure (loud RuntimeError), never a trigger for
            # re-running the already-executed graph prefix.
            if cut < len(self.records):
                self._direct_actions(self.records[cut:])

    def _capture(self, recs, cut: int, _segmented: bool) -> None:
        from nbxtensor.backend.replay_graph import CapturedPlan
        if os.environ.get("NBX_REPLAY_GRAPH_DUMP") == "1":
            self._dump_actions(recs)
        if _segmented:
            from nbxtensor.backend.replay_graph import SegmentedPlan
            self.graph = SegmentedPlan(recs, getattr(self.slab, "dev_idx", None))
            _log.debug(f"[Replay][GRAPH] bucket captured: {self.graph.launches} launches as "
                       f"{self.graph.stretches} CUDA graphs, one per stretch on one card")
        else:
            self.graph = CapturedPlan(
                recs, getattr(self.slab, "dev_idx", None))
            _log.debug(f"[Replay][GRAPH] bucket captured: "
                       f"{self.graph.launches} launches as one CUDA "
                       f"graph ({cut}/{len(self.records)} actions "
                       f"in-graph)")

    def _dump_actions(self, recs) -> None:
        from collections import Counter as _Ctr
        seqd = "".join(t for t, _ in recs[:60])
        names = [getattr(r[0], "name", "?")[:28]
                 for t, r in recs[:12] if t == KERNEL]
        cnt = _Ctr(t for t, _ in self.records)
        sizes = sorted({len(r[1]) for t, r in self.records
                        if t == H2D})
        _log.debug(f"[Replay][GRAPH] action tags[:60]={seqd} "
                   f"counts={dict(cnt)} h2d_sizes={sizes[:8]} "
                   f"first kernels={names}")
        tail = [(t, getattr(r[0], "name", "?")[:24]
                 if t == KERNEL else
                 (r[2] if t in (MEMCPY,) else len(r[1])
                  if t == H2D else r))
                for t, r in self.records[-16:]]
        _log.debug(f"[Replay][GRAPH] tail16={tail}")

    def _graph_refused(self, e: Exception) -> None:
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
        _log.debug(f"[Replay][GRAPH] refusal ({e}) — bucket keeps "
                   f"direct replay")
        self.graph_disabled = True
        if self.graph is not None:
            try:
                self.graph.destroy()
            finally:
                self.graph = None
        self._direct_actions()

    def hand_out_outputs(self, arena) -> None:
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
            arena[slot] = heap_copy(frozen)

    def crosses_cards(self) -> bool:
        """The recorded actions switch to a card other than the plan's own (`SETDEV`)."""
        try:
            return self._crosses
        except AttributeError:
            own = getattr(self.slab, "dev_idx", None)
            self._crosses = any(tag == SETDEV and rec[0] != own for tag, rec in self.records)
            return self._crosses

    def _direct_actions(self, records=None) -> None:
        """Phase 4a direct C-launcher replay of the action list."""
        if records is None:
            records = self.records
        for idx, (tag, rec) in enumerate(records):
            try:
                _run_action(tag, rec)
            except BaseException as e:
                name = (getattr(rec[0], "name", "?")
                        if tag == KERNEL else tag)
                raise RuntimeError(
                    f"replay action {idx}/{len(self.records)} "
                    f"({name}) failed: {e}") from e


def _run_action(tag, rec) -> None:
    """One recorded action, run directly."""
    if tag == SETDEV:
        # Triton keeps its OWN current device, so setting the
        # runtime's alone would still launch on the wrong card.
        DeviceAllocator.set_device(rec[0])
        DeviceAllocator.ensure_triton_device(rec[0])
    elif tag == KERNEL:
        kernel, g0, g1, g2, stream, flat = rec
        kernel.run(g0, g1, g2, stream, kernel.function,
                   kernel.packed_metadata, None, None, None,
                   *flat)
    elif tag == MEMCPY:
        DeviceAllocator.memcpy(*rec)
    elif tag == SYNC:
        _cur = DeviceAllocator.get_device()
        if _cur != rec[0]:
            DeviceAllocator.set_device(rec[0])
        DeviceAllocator.sync_device()
        if _cur != rec[0]:
            DeviceAllocator.set_device(_cur)
    elif tag == H2D:
        import ctypes
        dst, snap = rec
        buf = (ctypes.c_char * len(snap)).from_buffer_copy(snap)
        DeviceAllocator.memcpy(dst, ctypes.addressof(buf),
                               len(snap), 1)
    else:
        DeviceAllocator.memset_cuda(*rec)


# ---------------------------------------------------------------------------
# Orchestration — called from TritonSequence.run
# ---------------------------------------------------------------------------




def heap_copy(t: NBXTensor) -> NBXTensor:
    """A fresh heap tensor holding `t`'s LOGICAL values, contiguous. A graph output can be a view (a permute, a
    transpose): copying its storage bytes into a contiguous tensor of its shape scrambled it — Allegro-TI2V's encoder
    ends on a permute to frames-first, and every replayed tile of the I2V conditioning came back in the wrong order,
    the video a mosaic of tiles (2026-11-04)."""
    if t.is_contiguous():
        fresh = NBXTensor.empty_like(t)
        DeviceAllocator.memcpy(fresh.data_ptr(), t.data_ptr(), t.nbytes(), 3)
        return fresh
    return t.contiguous()                          # a strided copy kernel: the values in their logical order


def copy_into(frozen: NBXTensor, new: NBXTensor) -> None:
    """`new`'s values into the recorded `frozen` buffer, whatever either's layout (a view on either side is copied
    element by element, the recorded layout kept)."""
    if new.is_same_memory(frozen):
        return        # another view object of the recorded elements (a weight's view rebuilt after a request's bind)
    if new.is_contiguous() and frozen.is_contiguous():
        DeviceAllocator.memcpy(frozen.data_ptr(), new.data_ptr(), frozen.nbytes(), 3)
    else:
        frozen.copy_(new)


def _copy_slot(slot, frozen: NBXTensor, new: NBXTensor) -> None:
    """`copy_into`, a failure named with both buffers."""
    try:
        copy_into(frozen, new)
    except Exception as e:
        raise RuntimeError(
            f"replay input-copy failed at slot {slot}: shape "
            f"{tuple(frozen.shape)} dtype {frozen.nbx_dtype} "
            f"nbytes {frozen.nbytes()} frozen_dev="
            f"{getattr(frozen, '_device_idx', '?')} new_dev="
            f"{getattr(new, '_device_idx', '?')} frozen_ptr="
            f"{frozen.data_ptr():#x} new_ptr={new.data_ptr():#x}"
        ) from e


# --- A step's slab: measured, made room for, allocated, recorded into (the caller decides when) -----------------------


def measure(run, device_idx) -> Tuple[int, Dict[int, int], int]:
    """Run `run()` under the watermark counter: (the slab this card's recording needs — what the recording's first fit
    reaches over the measured run's allocations and frees, never less than the live peak —, {another card the run
    allocated on: its slab}, the live peak the verify pass holds)."""
    STATE.open_measurement()
    try:
        run()
    finally:
        STATE.measuring = False
    from nbxtensor.memory.slab import first_fit_extent
    extent = {d: first_fit_extent(ev) for d, ev in STATE.events_by_dev.items()}
    own = max(STATE.high_water, extent.get(device_idx, 0))
    others = {d: max(b, extent.get(d, 0)) for d, b in STATE.high_water_by_dev.items() if d != device_idx and b > 0}
    return own, others, int(STATE.high_water)


def room_for(dev: int, slab_bytes: int, verify_bytes: int) -> Tuple[bool, int]:
    """Give the card's cached blocks back to the driver (the pool's and the expandable range's free chunks), then
    whether a slab of `slab_bytes` and a verify pass of `verify_bytes` beside it fit what it has free: (fits, free)."""
    DeviceAllocator.give_back_cached(dev)
    free, _ = DeviceAllocator.mem_get_info(dev)
    return slab_bytes + verify_bytes <= free, free


def slabs(slab_bytes: int, device_idx, others: Dict[int, int]) -> Tuple[SlabAllocator, Dict[int, SlabAllocator]]:
    """The recording's slab on its card and one on each other card it allocates on; nothing held on a failure."""
    slab = SlabAllocator(slab_bytes, device_idx)
    try:
        return slab, {d: SlabAllocator(int(b), d) for d, b in (others or {}).items()}
    except Exception:
        slab.retire()
        raise


def record(run, slab: SlabAllocator, others: Dict[int, SlabAllocator]):
    """Run `run()` recorded into `slab` (and `others`): (the actions, the exception it raised or None, why the
    recording broke or None). Nothing of the recording state outlives the call."""
    STATE.recording = True
    arm_launch_recorder(True)
    STATE.records = []
    STATE.broken = None
    STATE.slab = slab
    STATE.slabs = dict(others)
    failed: Optional[BaseException] = None
    try:
        run()
    except BaseException as e:  # noqa: BLE001 — recording must not corrupt user runs
        failed = e
    finally:
        STATE.recording = False
        arm_launch_recorder(False)
        STATE.slab = None
        STATE.slabs = {}
        records = STATE.records
        STATE.records = []
    return records, failed, STATE.broken


def retire(plan: "FrozenPlan") -> None:
    """Release a plan's slabs and its captured graph (a plan rejected at its verify)."""
    plan.slab.retire()
    for _o in plan.other_slabs.values():
        _o.retire()
    if plan.graph is not None:
        # The verify pass may have captured the CUDA graph —
        # release exec/graph/staging/stream with the rejected
        # plan (gardien 2026-08-17: the drop leaked them).
        try:
            plan.graph.destroy()
        finally:
            plan.graph = None


def head_values(t: NBXTensor, nbytes: int = 64):
    """The first `nbytes` of `t`'s memory read to the host as float16 values (a diagnostic's look)."""
    import ctypes as _c
    import numpy as _np
    nb = min(t.nbytes(), nbytes)
    rb = (_c.c_char * nb)()
    DeviceAllocator.memcpy(_c.addressof(rb), t.data_ptr(), nb, 2)
    return _np.frombuffer(bytes(rb), dtype=_np.float16)

