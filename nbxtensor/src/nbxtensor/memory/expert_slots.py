"""Device slots for the routed experts of a mixture-of-experts component: the arena, its copies, the card's own map.

A component whose experts do not all fit on its card keeps them on page-locked host memory and a cache of `n_slots`
(gate, up, down) device triples at the experts' own shape and dtype — in ONE device allocation, the arena, carved into
256-byte-aligned views: a driver rounds every allocation to its own granule (2 MiB on CUDA, measured 2026-10-14: a
3 MiB block costs 4 MiB), and 2 163 slots as 6 489 separate 3 MiB mallocs carried 6.3 GB no table knew about. One
arena is rounded once, so the store's device bytes are the caller's figure to within one granule, on every platform.

WHICH expert sits in WHICH slot is not decided here. The caller hands a `residency` — any object that answers:

    n_slots                              the slot count (the caller's plan)
    assign(layer, routed)             -> ({expert: slot}, [experts that missed])
    reserve(layer, experts, protect)  -> {expert: slot} of a copy made ahead of its call
    resident()                        -> [((layer, expert), slot)], least recently routed first
    reset() / clear()                    every slot free again
    hits, misses, summary()              its counters (the in-place store counts into them)

and this store moves the bytes: `acquire` copies every miss into its slot with `memcpy_async` on the store's stream —
after that stream waited for the compute stream, so a slot is rewritten only once the kernels reading its former
expert have run — and makes the compute stream wait for the copies. It returns the [E] weight lists a pointer table is
built from: a routed expert reads its slot, an unrouted one points at a routed expert's slot (a valid address the
kernel never dereferences — it loads `expert_ptrs[e]` only for the experts the aligned blocks name).

The allocator's stream is BLOCKING against the legacy stream (`create_stream`), so the copies queue between kernels
rather than beside them: the gain is the cache (only misses cross the link, once), not overlap.

For one decode token the same least-recently-used rule runs ON THE CARD (`enable_device_decode`, `decode_on_card`,
the kernels of `ops.moe_device_residency`): a slot map per (layer, expert), each slot's owner and last-use stamp, and a
clock the assign kernel advances at every call. The host's map and the card's never both hold the slots: whichever
runs after the other first takes the other's state (`_sync_device_from_host`, `_host_takes_over`), and the host knows
the card moved its slots by reading the card's clock — a replayed step runs no host code.

`HeldExperts` is the other holding: every expert of whole layers copied once onto ANOTHER card, in one arena there.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Dict, Tuple

from nbxtensor.dtype.nbx_dtype import NBXDtype, dtype_size
from nbxtensor.memory.allocator import DeviceAllocator as DA
from nbxtensor.memory.layout import aligned
from nbxtensor.shape import numel
from nbxtensor.tensor import NBXTensor


class ExpertSlotStore:
    """`residency.n_slots` device slots of one expert geometry (`like`: one expert's (gate, up, down) host triple) on
    device `device_idx`. `in_place`: on a unified device whose experts are views of mapped files, a slot is a POINTER
    to the expert where it lies — its pages bound in place (`DeviceAllocator.bind_mapped`), nothing copied, no arena."""

    def __init__(self, residency, device_idx: int, like, in_place: bool = False):
        from nbxtensor.memory.pointer_tables import TablePins
        self.residency = residency
        self.device_idx = int(device_idx)
        self.in_place = bool(in_place)
        # (layer, expert) -> its projections' host addresses, least recently routed first. Bounded by the slot count:
        # a bound expert's pages are resident while a command binds them, so the bound set is the cache the caller
        # priced — the in-place store never holds more than the copying one (unbounded, 2026-10-20: 27 GB of experts
        # bound in one prefill, swap +6.7 GB in 13 s).
        self._bound = OrderedDict()
        self.table_pins = TablePins()     # its calls' table pins, ended before an unbind and at release
        for t in like:
            if not isinstance(t, NBXTensor):
                raise RuntimeError(
                    f"ZERO FALLBACK: the expert store copies dense NBXTensor experts; got {type(t).__name__} "
                    "(a quantized expert carries packed weights + scales — not this store's geometry)")
        self._shape = tuple((tuple(t.shape), t.nbx_dtype) for t in like)
        DA.set_device(self.device_idx)
        self._stream = 0
        self.bytes_in = 0
        self._inflight = {}      # (layer, expert) -> the event of its prefetch copy
        self._arena = None
        self.slots = []
        self.device_decode = False    # the residency on the card for a decode token (`enable_device_decode`)
        if not self.in_place:
            self._carve_slots()
            self._stream = DA.create_stream()

    def _carve_slots(self) -> None:
        """The arena: one allocation for every slot, each projection a 256-byte-aligned view into it."""
        proj_bytes = [aligned(numel(s) * dtype_size(d)) for s, d in self._shape]
        stride = sum(proj_bytes)
        self._arena = NBXTensor.empty((stride * self.residency.n_slots,), dtype=NBXDtype.uint8,
                                      device=f"cuda:{self.device_idx}")
        base = self._arena.data_ptr()
        for k in range(self.residency.n_slots):
            off = base + k * stride
            views = []
            for (s, d), nb in zip(self._shape, proj_bytes):
                views.append(NBXTensor.from_raw(off, s, d, "cuda", owns_data=False,
                                                device_idx=self.device_idx, base=self._arena))
                off += nb
            self.slots.append(tuple(views))

    # ── the residency on the card, for a decode token (`ops/moe_device_residency.py`) ──
    def enable_device_decode(self, layer_index: Dict, n_experts: int) -> None:
        """Arms the store's residency ON THE CARD for one-token calls: a slot map per (layer, expert), each slot's
        owner and last-use stamp, a clock, the slots' fixed pointer tables, and (filled per layer at its first call)
        the device-visible addresses of every host expert. `layer_index`: {layer key: index} of the component's
        store-served layers. The host residency and this one never both hold the slots: whichever runs after the
        other first takes the other's state (`_sync_device_from_host`, `_host_takes_over`)."""
        import numpy as np
        L, E, S = len(layer_index), int(n_experts), int(self.residency.n_slots)
        DA.set_device(self.device_idx)
        dev = self.device_idx
        self.layer_index, self.n_experts = dict(layer_index), E
        self._slot_of = NBXTensor.from_numpy(np.full(L * E, -1, dtype=np.int32), device=dev)
        self._owner = NBXTensor.from_numpy(np.full(S, -1, dtype=np.int32), device=dev)
        self._stamp = NBXTensor.from_numpy(np.full(S, -1, dtype=np.int64), device=dev)
        self._clock = NBXTensor.from_numpy(np.zeros(1, dtype=np.int64), device=dev)
        self._slot_tabs = tuple(NBXTensor.from_numpy(np.array([self.slots[k][i].data_ptr() for k in range(S)],
                                                              dtype=np.int64), device=dev) for i in range(3))
        self._host_np = np.zeros((3, L * E), dtype=np.int64)
        self._host_tabs = None
        self._host_layers = set()
        self._dev_valid = False       # the card's map reflects the slots' contents
        self._host_dirty = False      # the card's residency moved the slots since the host's last call (a direct step)
        self._clock_seen = 0          # the card's clock when the host last wrote the map (`_card_moved_slots`)
        self.device_decode = True
        self._copy = None             # the miss copy's (block, warps), read once from the profile (`_copy_launch`)

    def _host_tables_for(self, layer, host: Callable[[int], Tuple]) -> None:
        if layer in self._host_layers:
            return
        li = self.layer_index[layer]
        for e in range(self.n_experts):
            for i, t in enumerate(host(e)):
                if (not isinstance(t, NBXTensor) or t.raw_device != "cpu" or not t.is_contiguous()
                        or (tuple(t.shape), t.nbx_dtype) != self._shape[i] or not t.is_pinned):
                    raise RuntimeError(
                        f"ZERO FALLBACK: expert {e} of {layer!r} projection {i} is not a contiguous page-locked host "
                        f"expert of {self._shape[i]} — the card reads its misses from page-locked host memory")
                self._host_np[i, li * self.n_experts + e] = t.data_ptr()
        self._host_tabs = tuple(NBXTensor.from_numpy(self._host_np[i], device=self.device_idx) for i in range(3))
        self._host_layers.add(layer)

    def _sync_device_from_host(self) -> None:
        """The card's map rebuilt from the host residency (its least-recently-used order becomes the stamps), after
        every host prefetch still copying has landed: the card then sees exactly which expert each slot holds."""
        import numpy as np
        for (key, ev) in list(self._inflight.items()):
            DA.stream_wait_event(0, ev, flags=0)
            DA.destroy_event(ev)
        self._inflight = {}
        L, E, S = len(self.layer_index), self.n_experts, self.residency.n_slots
        slot_of = np.full(L * E, -1, dtype=np.int32)
        owner = np.full(S, -1, dtype=np.int32)
        stamp = np.full(S, -1, dtype=np.int64)
        n = 0
        for (layer, e), slot in self.residency.resident():
            li = self.layer_index.get(layer)
            if li is None:
                continue
            slot_of[li * E + int(e)] = slot
            owner[slot] = li * E + int(e)
            stamp[slot] = n
            n += 1
        DA.set_device(self.device_idx)
        for dst, src in ((self._slot_of, slot_of), (self._owner, owner), (self._stamp, stamp),
                         (self._clock, np.array([n], dtype=np.int64))):
            DA.memcpy(dst.data_ptr(), src.ctypes.data, src.nbytes, 1)
        self._clock_seen = n
        self._dev_valid = True

    def _card_moved_slots(self) -> bool:
        """Whether the card's residency assigned slots since the host last wrote its map: the card's own clock moved
        (`expert_slot_assign_kernel` advances it at every call). A REPLAYED decode step runs no host code, so
        `_host_dirty` (set by a direct step) never sees it: on a served mixture-of-experts model the host then
        prefilled a turn through slots the replayed decode had refilled with other experts, and wrote that stale map
        back to the card — the session's answers went wrong from the second repeated turn on (2026-11-12)."""
        if not self._dev_valid:
            return False
        import numpy as np
        now = np.zeros(1, dtype=np.int64)
        DA.set_device(self.device_idx)
        DA.sync_device()                  # the replayed steps' assignments land before the clock is read
        DA.memcpy(now.ctypes.data, self._clock.data_ptr(), now.nbytes, 2)
        return int(now[0]) != self._clock_seen

    def _host_takes_over(self) -> None:
        """The host residency resumes after the card's: what the card put in the slots is not in the host's map, so
        the host starts from empty slots (its next calls miss and copy), and the card's map is rebuilt from the host's
        at its next call."""
        if self._host_dirty or self._card_moved_slots():
            self.residency.reset()
            self._host_dirty = False
        self._dev_valid = False

    # ── what a fused call (`ops.moe_fused.fused_moe`) asks and tells the store; a subclass decides more ──
    def decodes_on_card(self, layer) -> bool:
        """Whether a one-token call at `layer` runs the residency on the card (`decode_on_card`): armed, and `layer`
        one of the card's map."""
        return self.device_decode and layer in self.layer_index

    def on_routed(self, layer, routed, host: Callable[[int], Tuple]) -> None:
        """A call at `layer` read back the experts it routes to (`routed`, sorted) before its slots are acquired;
        `host(e)` gives expert e's host triple. The store keeps nothing of it."""

    def on_enqueued(self, layer, routed) -> None:
        """A grouped call at `layer` has its launches enqueued, its slots acquired for `routed`. Nothing here."""

    def decode_on_card(self, layer, flat_indices, flat_scores, hidden, host, top_k: int, K: int, N_gate: int):
        """One token through a store-served layer with the residency on the card: assign the routed experts' slots
        (misses evicting the least recently used), copy the misses from page-locked host memory, run the half decode
        band through the slots' pointer table. No host decision: every launch is the same at every token."""
        import triton as _tr
        from nbxtensor.ops.moe_device_residency import expert_slot_assign_kernel, expert_miss_copy_kernel
        self._host_tables_for(layer, host)
        if not self._dev_valid:
            self._sync_device_from_host()
        dev = self.device_idx
        DA.set_device(dev)
        DA.ensure_triton_device(dev)
        S = self.residency.n_slots
        base = self.layer_index[layer] * self.n_experts
        slot_ids = NBXTensor.empty((top_k,), dtype=NBXDtype.int64, device=f"cuda:{dev}")
        miss = NBXTensor.empty((top_k,), dtype=NBXDtype.int64, device=f"cuda:{dev}")
        ids = flat_indices if flat_indices.nbx_dtype == NBXDtype.int64 else flat_indices.to(NBXDtype.int64)
        expert_slot_assign_kernel[(1,)](ids, self._slot_of, self._owner, self._stamp, self._clock, slot_ids, miss,
                                        base, S, TOP_K=top_k, BLOCK_S=_tr.next_power_of_2(S), num_warps=4)
        n = [numel(sh) for sh, _d in self._shape]
        bf16 = self._shape[0][1] == NBXDtype.bfloat16
        block, warps = self._copy_launch()
        expert_miss_copy_kernel[(top_k, 3, _tr.cdiv(max(n), block))](
            miss, slot_ids, *self._host_tabs, *self._slot_tabs, base, n[0], n[1], n[2], BLOCK=block, W_BF16=bf16,
            num_warps=warps)
        self._host_dirty = True
        return self._band_through_slots(hidden, slot_ids, flat_scores, top_k, K, N_gate, bf16)

    def _band_through_slots(self, hidden, slot_ids, flat_scores, top_k: int, K: int, N_gate: int, bf16: bool):
        """The half decode band read through the slots' fixed pointer tables, indexed by slot."""
        from types import SimpleNamespace
        from nbxtensor.ops.moe_wrappers import moe_decode_vec_half_pass
        dev, s0 = self.device_idx, self.slots[0]
        tables = SimpleNamespace(
            gate_ptrs={dev: self._slot_tabs[0]}, up_ptrs={dev: self._slot_tabs[1]}, down_ptrs={dev: self._slot_tabs[2]},
            gate_stride_bk={dev: s0[0].stride(1)}, gate_stride_bn={dev: s0[0].stride(0)},
            down_stride_bk={dev: s0[2].stride(1)}, down_stride_bn={dev: s0[2].stride(0)})
        return moe_decode_vec_half_pass(hidden, tables, dev, slot_ids, flat_scores, top_k, K, N_gate, bf16)

    def _copy_launch(self):
        """(block, warps) of the miss copy on this store's card: the profile's measured `expert_copy` (measured at the
        first launch by `nbxtensor.ops.moe_device_residency.measure_miss_copy`, autodetect.ensure_measured_facts) — refused by name when
        the profile does not carry it, never a launch carried from another card."""
        if self._copy is None:
            from nbxtensor.ops.wrappers import get_hardware_profile
            prof = get_hardware_profile()
            devs = list(getattr(prof, "devices", None) or [])
            dev = next((d for d in devs if getattr(d, "index", None) == self.device_idx), None)
            fact = (getattr(dev, "measured", None) or {}).get("expert_copy") if dev is not None else None
            if not fact or "block" not in fact or "warps" not in fact:
                raise RuntimeError(
                    f"ZERO FALLBACK: the expert store copies its misses with the launch measured on its card "
                    f"(the profile's measured.expert_copy: nbxtensor.ops.moe_device_residency.measure_miss_copy, written at the first "
                    f"Triton launch by autodetect.ensure_measured_facts), and the profile "
                    f"{getattr(prof, 'id', None)!r} carries none for device {self.device_idx}")
            self._copy = (int(fact["block"]), int(fact["warps"]))
        return self._copy

    @property
    def slot_bytes(self) -> int:
        return sum(t.nbytes() for t in self.slots[0]) if self.slots else 0

    @property
    def device_bytes(self) -> int:
        """What the store holds on the device: the arena, one allocation (the driver rounds it once)."""
        return self._arena.nbytes() if self._arena is not None else 0

    def acquire(self, layer, routed, n_experts: int, host: Callable[[int], Tuple]):
        """The host residency's call (a prompt's tokens, or any call the card's residency does not serve). With the
        card's residency armed, the card's slot map is rewritten from the host's at the end of the call: a replayed
        decode step runs no host code, so the map it reads must already say what the host just put in the slots."""
        out = self._acquire(layer, routed, n_experts, host)
        if self.device_decode:
            self._sync_device_from_host()
        return out

    def _acquire(self, layer, routed, n_experts: int, host: Callable[[int], Tuple]):
        """(gate, up, down) lists over all `n_experts`, device-resident, for a call routing to `routed`;
        the compute stream waits for the misses copied now and for the hits a prefetch is still copying.
        In place: the routed experts' own host views, their pages bound once; nothing is copied."""
        if self.device_decode:
            self._host_takes_over()
        if self.in_place:
            return self._acquire_in_place(layer, routed, n_experts, host)
        slots, missed = self.residency.assign(layer, routed)
        for e in slots:
            ev = self._inflight.pop((layer, e), None)
            if ev is not None:
                DA.stream_wait_event(0, ev, flags=0)
                DA.destroy_event(ev)
        if missed:
            DA.set_device(self.device_idx)
            self._store_stream_waits_for_compute()
            for e in missed:
                self._copy_in(layer, e, self.slots[slots[e]], host(e))
            ev = DA.create_event(timing=False)
            DA.record_event(ev, stream=self._stream)
            DA.stream_wait_event(0, ev, flags=0)
            DA.destroy_event(ev)
        filler = self.slots[slots[min(slots)]]
        triples = [self.slots[slots[e]] if e in slots else filler for e in range(int(n_experts))]
        return ([t[0] for t in triples], [t[1] for t in triples], [t[2] for t in triples])

    def _store_stream_waits_for_compute(self) -> None:
        """The store's stream waits for what the compute stream holds now: a slot is rewritten only once the kernels
        reading its former expert have run."""
        ev = DA.create_event(timing=False)
        DA.record_event(ev, stream=0)
        DA.stream_wait_event(self._stream, ev, flags=0)
        DA.destroy_event(ev)

    def _refuse_unless_host_expert(self, layer, e: int, i: int, s, does: str) -> None:
        """One geometry per component: projection `i` of expert `e` is a contiguous host tensor of the store's shape
        and dtype, or the call is refused by name (`does`: what this store does with a host expert)."""
        if (not isinstance(s, NBXTensor) or (tuple(s.shape), s.nbx_dtype) != self._shape[i]
                or not s.is_contiguous() or s.raw_device != "cpu"):
            raise RuntimeError(
                f"ZERO FALLBACK: expert {e} of {layer!r} projection {i} is "
                f"{type(s).__name__} {tuple(getattr(s, 'shape', ()))} on "
                f"{getattr(s, '_device', '?')}; {does} of "
                f"{self._shape[i]} — one geometry per component")

    def _copy_in(self, layer, e: int, dst, src) -> None:
        """Expert `e`'s host projections copied into the slot `dst`, on the store's stream."""
        for i, (d, s) in enumerate(zip(dst, src)):
            self._refuse_unless_host_expert(layer, e, i, s, "the store copies contiguous host experts")
            DA.memcpy_async(d.data_ptr(), s.data_ptr(), s.nbytes(), kind=1, stream=self._stream)
            self.bytes_in += s.nbytes()

    def _acquire_in_place(self, layer, routed, n_experts: int, host: Callable[[int], Tuple]):
        views = {}
        for e in sorted(int(x) for x in routed):
            src = host(e)
            for i, s in enumerate(src):
                self._refuse_unless_host_expert(layer, e, i, s, "the in-place store binds contiguous host views")
            if (layer, e) not in self._bound:
                self._bound[(layer, e)] = ([s.data_ptr() for s in src], self._bind(src))
                self.residency.misses += 1
            else:
                self._bound.move_to_end((layer, e))
                self.residency.hits += 1
            views[e] = self._bound[(layer, e)][1]
        self._evict_bound(keep={(layer, e) for e in views})
        filler = views[min(views)]
        triples = [views.get(e, filler) for e in range(int(n_experts))]
        return ([t[0] for t in triples], [t[1] for t in triples], [t[2] for t in triples])

    def _bind(self, src) -> Tuple:
        """The DEVICE views of an expert's host projections, their pages bound in place: a pointer table captures a
        device address only from a device tensor — a host view's address read as zeros (2026-10-20)."""
        dev = []
        for s in src:
            DA.bind_mapped(s.data_ptr(), s.nbytes())
            dev.append(NBXTensor.from_raw(s.data_ptr(), tuple(s.shape), s.nbx_dtype, "cuda", owns_data=False,
                                          device_idx=self.device_idx, base=s))
        return tuple(dev)

    def _evict_bound(self, keep) -> None:
        """Unbind the least recently routed experts past the slot count (never one this call reads), pins ended first."""
        cap = max(int(self.residency.n_slots), len(keep))
        if len(self._bound) <= cap:
            return
        self.table_pins.end(cached=True)
        for key in [k for k in self._bound if k not in keep][:len(self._bound) - cap]:
            for ptr in self._bound.pop(key)[0]:
                DA.unbind_mapped(ptr)

    def prefetch(self, layer, experts, host: Callable[[int], Tuple], protect) -> int:
        """Copy `experts` of `layer` ahead of their call, on the store's stream, into slots outside `protect` (the
        keys another call still reads); the number copied. Their call waits for each copy's event."""
        n = self._prefetch(layer, experts, host, protect)
        if self.device_decode and n:
            self._sync_device_from_host()
        return n

    def _prefetch(self, layer, experts, host: Callable[[int], Tuple], protect) -> int:
        if self.device_decode:
            self._host_takes_over()
        slots = self.residency.reserve(layer, experts, protect)
        if not slots:
            return 0
        DA.set_device(self.device_idx)
        self._store_stream_waits_for_compute()
        for e, k in slots.items():
            self._copy_in(layer, e, self.slots[k], host(e))
            ev = DA.create_event(timing=False)
            DA.record_event(ev, stream=self._stream)
            self._inflight[(layer, e)] = ev
        return len(slots)

    def summary(self) -> str:
        return f"[expert store] {self.residency.summary()} bytes_in={self.bytes_in / 1e9:.2f} GB"

    def release(self) -> None:
        for ev in self._inflight.values():
            DA.destroy_event(ev)
        self._inflight.clear()
        self.table_pins.end()
        DA.set_device(self.device_idx)
        if self._stream:
            DA.stream_synchronize(self._stream)
            DA.destroy_stream(self._stream)
        self._stream = 0
        self.slots = []
        self._arena = None
        self.residency.clear()


class HeldExperts:
    """The routed experts of whole layers held on ANOTHER card: every expert of each held layer copied once, at
    construction, into one arena on that card — each projection a 256-byte-aligned view, the slot store's own layout —
    so the layer's fused op runs there against resident weights (its hidden state crosses to the card and back).
    `layers` maps a layer key to its host expert triples [(gate, up, down)] in expert order."""

    def __init__(self, device_idx: int, layers):
        self.device_idx = int(device_idx)
        self.layers = {}
        total = self._host_bytes(layers)
        DA.set_device(self.device_idx)
        # one allocation per card, outside the pool (a long-lived block a pool would let activations split once
        # freed): it lives as long as the component's store
        self._arena = NBXTensor.from_raw(DA.malloc_cuda(total, pooled=False), (total,), NBXDtype.uint8, "cuda",
                                         owns_data=True, device_idx=self.device_idx)
        off = self._arena.data_ptr()
        for key, triples in layers.items():
            g, u, d = [], [], []
            for trip in triples:
                views, off = self._copy_expert(trip, off)
                g.append(views[0]); u.append(views[1]); d.append(views[2])
            self.layers[key] = (g, u, d)
        DA.sync_device()
        self.bytes = total

    def _host_bytes(self, layers) -> int:
        """The arena's size: every projection of every held expert at its aligned size — each a contiguous host
        tensor, or the holding is refused by name."""
        total = 0
        for key, triples in layers.items():
            for t in (t for trip in triples for t in trip):
                if not isinstance(t, NBXTensor) or t.raw_device != "cpu" or not t.is_contiguous():
                    raise RuntimeError(
                        f"ZERO FALLBACK: an expert of {key!r} held on card {self.device_idx} is "
                        f"{type(t).__name__} on {getattr(t, '_device', '?')}; the card is filled from contiguous "
                        f"host experts")
                total += aligned(t.nbytes())
        return total

    def _copy_expert(self, trip, off: int):
        """One expert's projections copied to the arena from address `off`: (its device views, the next address)."""
        views = []
        for t in trip:
            v = NBXTensor.from_raw(off, tuple(t.shape), t.nbx_dtype, "cuda", owns_data=False,
                                   device_idx=self.device_idx, base=self._arena)
            DA.memcpy(v.data_ptr(), t.data_ptr(), t.nbytes(), 1)
            views.append(v)
            off += aligned(t.nbytes())
        return views, off

    def release(self) -> None:
        self.layers = {}
        self._arena = None
