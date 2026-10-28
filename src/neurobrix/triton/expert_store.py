"""The triton engine's expert store: device slots for routed experts, filled on a transfer stream.

Mode 2 (NBXTensor, zero torch — R33). The compiled engine has its own
(`core/runtime/graph/expert_store.py`); the two share the residency POLICY
(`core/runtime/expert_residency.py`, no tensor library) and no compute code.

The component's expert weights stay on pinned host memory; the store preallocates `n_slots`
(gate, up, down) device triples at the experts' own shape and dtype — in ONE device allocation, the
arena, carved into 256-byte-aligned views: a driver rounds every allocation to its own granule (2 MiB
on CUDA, measured on the rig 2026-10-14: a 3 MiB block costs 4 MiB), and 2 163 slots as 6 489 separate
3 MiB mallocs carried 6.3 GB no table of the engine knew about (Qwen3-Omni-30B-A3B-Instruct, 32 GB
V100: 31.6 GB used for 24.8 GB allocated). One arena is rounded once, so the store's device bytes are
Prism's figure to within one granule, on every platform, with no number of any platform written here.
`execute_moe_fused` reads the
routed experts once on the host, then `acquire` copies every miss into its slot with
`memcpy_async` on the store's stream — after that stream waited for the compute stream, so a slot
is rewritten only once the kernels reading its former expert have run — and makes the compute
stream wait for the copies before the grouped GEMM. It returns the [E] weight lists the pointer
tables are built from: a routed expert reads its slot, an unrouted one points at a routed
expert's slot (a valid address the kernel never dereferences — it loads `expert_ptrs[e]` only for
the experts the aligned blocks name).

The DeviceAllocator stream is BLOCKING against the legacy stream (`create_stream`), so the copies
queue between kernels rather than beside them: the gain here is the cache (only misses cross the
link, once), not overlap.
"""
from __future__ import annotations

import os
from typing import Callable, List, Tuple

from neurobrix.core.runtime.expert_residency import ExpertResidency
from neurobrix.kernels.nbx_tensor import DeviceAllocator as DA, NBXDtype, NBXTensor, dtype_size

_ALIGN = 256   # every projection starts on a 256-byte boundary (the widest vector load of any backend)


def _aligned(nbytes: int) -> int:
    return -(-int(nbytes) // _ALIGN) * _ALIGN


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


class NBXExpertStore:
    def __init__(self, residency: ExpertResidency, device_idx: int, like, in_place: bool = False):
        self.residency = residency
        self.device_idx = int(device_idx)
        self.trace = None     # an ExpertTrace (core/runtime/routing_reuse.py) the store notes each decode call into
        # In place (supervisor 2026-10-20 01:20 (b)): on a unified device whose routed experts are views of the
        # container's files (Prism's `expert_files`, `holds_expert_files`), a slot is a POINTER to the expert
        # where it lies — its pages bound in place (`DeviceAllocator.bind_mapped`), nothing copied, no arena.
        self.in_place = bool(in_place)
        # (layer, expert) -> its projections' host addresses, least recently routed first. Bounded by the plan's
        # slot count: a bound expert's pages are resident while a command binds them, so the bound set is the
        # cache Prism priced — the in-place store never holds more than the copying one (DSV2 triton unbounded,
        # 2026-10-20: 27 GB of experts bound in the prefill, swap +6.7 GB in 13 s).
        from collections import OrderedDict
        self._bound = OrderedDict()
        for t in like:
            if not isinstance(t, NBXTensor):
                raise RuntimeError(
                    f"ZERO FALLBACK: the expert store copies dense NBXTensor experts; got {type(t).__name__} "
                    "(a quantized expert carries packed weights + scales — not this store's geometry)")
        self._shape = tuple((tuple(t.shape), t._dtype) for t in like)
        DA.set_device(self.device_idx)
        self._stream = 0
        self.bytes_in = 0
        self._inflight = {}
        self.hosts = {}
        self._arena = None
        self.slots = []
        if self.in_place:
            return
        # The arena: one allocation for every slot, each projection a 256-byte-aligned view into it.
        proj_bytes = [_aligned(_numel(s) * dtype_size(d)) for s, d in self._shape]
        stride = sum(proj_bytes)
        self._arena = NBXTensor.empty((stride * residency.n_slots,), dtype=NBXDtype.uint8,
                                      device=f"cuda:{self.device_idx}")
        base = self._arena.data_ptr()
        self.slots: List[Tuple[NBXTensor, NBXTensor, NBXTensor]] = []
        for k in range(residency.n_slots):
            off = base + k * stride
            views = []
            for (s, d), nb in zip(self._shape, proj_bytes):
                views.append(NBXTensor.from_raw(off, s, d, "cuda", owns_data=False,
                                                device_idx=self.device_idx, base=self._arena))
                off += nb
            self.slots.append(tuple(views))
        self._stream = DA.create_stream()
        self.bytes_in = 0
        self._inflight = {}      # (layer, expert) -> the event of its prefetch copy
        self.hosts = {}          # layer -> host expert reader, registered by its call

    @property
    def slot_bytes(self) -> int:
        return sum(t._nbytes for t in self.slots[0]) if self.slots else 0

    @property
    def device_bytes(self) -> int:
        """What the store holds on the device: the arena, one allocation (the driver rounds it once)."""
        return self._arena._nbytes if self._arena is not None else 0

    def acquire(self, layer, routed, n_experts: int, host: Callable[[int], Tuple]):
        """(gate, up, down) lists over all `n_experts`, device-resident, for a call routing to `routed`;
        the compute stream waits for the misses copied now and for the hits a prefetch is still copying.
        In place: the routed experts' own host views, their pages bound once; nothing is copied."""
        if self.trace is not None:
            self.trace.note(layer, routed)      # the routing reuse curve (core/runtime/routing_reuse.py)
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
            ev = DA.create_event(timing=False)
            DA.record_event(ev, stream=0)
            DA.stream_wait_event(self._stream, ev, flags=0)
            DA.destroy_event(ev)
            for e in missed:
                dst = self.slots[slots[e]]
                src = host(e)
                for i, (d, s) in enumerate(zip(dst, src)):
                    if (not isinstance(s, NBXTensor) or (tuple(s.shape), s._dtype) != self._shape[i]
                            or not s.is_contiguous() or s._device != "cpu"):
                        raise RuntimeError(
                            f"ZERO FALLBACK: expert {e} of {layer!r} projection {i} is "
                            f"{type(s).__name__} {tuple(getattr(s, 'shape', ()))} on "
                            f"{getattr(s, '_device', '?')}; the store copies contiguous host experts of "
                            f"{self._shape[i]} — one geometry per component")
                    DA.memcpy_async(d.data_ptr(), s.data_ptr(), s._nbytes, kind=1, stream=self._stream)
                    self.bytes_in += s._nbytes
            ev = DA.create_event(timing=False)
            DA.record_event(ev, stream=self._stream)
            DA.stream_wait_event(0, ev, flags=0)
            DA.destroy_event(ev)
        filler = self.slots[slots[min(slots)]]
        triples = [self.slots[slots[e]] if e in slots else filler for e in range(int(n_experts))]
        return ([t[0] for t in triples], [t[1] for t in triples], [t[2] for t in triples])

    def _acquire_in_place(self, layer, routed, n_experts: int, host: Callable[[int], Tuple]):
        views = {}
        for e in sorted(int(x) for x in routed):
            src = host(e)
            for i, s in enumerate(src):
                if (not isinstance(s, NBXTensor) or (tuple(s.shape), s._dtype) != self._shape[i]
                        or not s.is_contiguous() or s._device != "cpu"):
                    raise RuntimeError(
                        f"ZERO FALLBACK: expert {e} of {layer!r} projection {i} is {type(s).__name__} "
                        f"{tuple(getattr(s, 'shape', ()))} on {getattr(s, '_device', '?')}; the in-place store "
                        f"binds contiguous host views of {self._shape[i]} — one geometry per component")
            if (layer, e) not in self._bound:
                dev = []
                for s in src:
                    DA.bind_mapped(s.data_ptr(), s._nbytes)
                    # the DEVICE view of the bound pages: the pointer tables capture a device address only from a
                    # device tensor — a host view's address read as zeros (DSV2 triton printed "!!!", 2026-10-20)
                    dev.append(NBXTensor.from_raw(s.data_ptr(), tuple(s.shape), s._dtype, "cuda", owns_data=False,
                                                  device_idx=self.device_idx, base=s))
                self._bound[(layer, e)] = ([s.data_ptr() for s in src], tuple(dev))
                self.residency.misses += 1
            else:
                self._bound.move_to_end((layer, e))
                self.residency.hits += 1
            views[e] = self._bound[(layer, e)][1]
        self._evict_bound(keep={(layer, e) for e in views})
        filler = views[min(views)]
        triples = [views.get(e, filler) for e in range(int(n_experts))]
        return ([t[0] for t in triples], [t[1] for t in triples], [t[2] for t in triples])

    def _evict_bound(self, keep) -> None:
        """Unbind the least recently routed experts past the slot count, never one this call reads. The cached
        pointer tables pin what they address, so they are dropped first (rebuilt by the next call)."""
        cap = max(int(self.residency.n_slots), len(keep))
        if len(self._bound) <= cap:
            return
        from neurobrix.triton.moe import release_pinned_tables
        release_pinned_tables()
        for key in [k for k in self._bound if k not in keep][:len(self._bound) - cap]:
            for ptr in self._bound.pop(key)[0]:
                DA.unbind_mapped(ptr)

    # ── step 3: the prefetch of the next layer's predicted experts (mirror of the compiled store) ──
    def prefetch_next(self, layer, routed) -> int:
        """Copy ahead, on the store's stream, the experts the previous token routed to at the layer that
        follows `layer`, into slots the call at `layer` does not hold; 0 when nothing can be planned
        or unless NBX_EXPERT_PREFETCH=1 asks for it (opt-in since 2026-10-19: 21 % right at decode on Ming,
        +23 % time on the Triton engine). Called once the call's kernels are
        enqueued: the stream is blocking against the legacy stream, so copies issued after the launch
        run behind it and ahead of the next layer's."""
        if os.environ.get("NBX_EXPERT_PREFETCH") != "1" or self.in_place:
            return 0
        nxt = self.residency.next_layer(layer)
        if nxt is None or nxt not in self.hosts:
            return 0
        protect = {(layer, int(e)) for e in routed}
        experts = self.residency.plan_prefetch(nxt, protect)
        if not experts:
            return 0
        return self.prefetch(nxt, experts, self.hosts[nxt], protect)

    def prefetch(self, layer, experts, host: Callable[[int], Tuple], protect) -> int:
        slots = self.residency.reserve(layer, experts, protect)
        if not slots:
            return 0
        DA.set_device(self.device_idx)
        ev = DA.create_event(timing=False)
        DA.record_event(ev, stream=0)
        DA.stream_wait_event(self._stream, ev, flags=0)
        DA.destroy_event(ev)
        for e, k in slots.items():
            dst = self.slots[k]
            src = host(e)
            for i, (d, s_) in enumerate(zip(dst, src)):
                if (not isinstance(s_, NBXTensor) or (tuple(s_.shape), s_._dtype) != self._shape[i]
                        or not s_.is_contiguous() or s_._device != "cpu"):
                    raise RuntimeError(
                        f"ZERO FALLBACK: expert {e} of {layer!r} projection {i} is "
                        f"{type(s_).__name__} {tuple(getattr(s_, 'shape', ()))} on "
                        f"{getattr(s_, '_device', '?')}; the store copies contiguous host experts of "
                        f"{self._shape[i]} — one geometry per component")
                DA.memcpy_async(d.data_ptr(), s_.data_ptr(), s_._nbytes, kind=1, stream=self._stream)
                self.bytes_in += s_._nbytes
            ev = DA.create_event(timing=False)
            DA.record_event(ev, stream=self._stream)
            self._inflight[(layer, e)] = ev
        return len(slots)

    def summary(self) -> str:
        return f"[expert store] {self.residency.summary()} bytes_in={self.bytes_in / 1e9:.2f} GB"

    def release(self) -> None:
        if self.trace is not None:
            self.trace.write(f"recorded by the triton expert store's decode ({self.residency.summary()})")
            self.trace = None
        if self.residency.hits or self.residency.misses:
            print(f"   {self.summary()}", flush=True)
        for ev in self._inflight.values():
            DA.destroy_event(ev)
        self._inflight.clear()
        DA.set_device(self.device_idx)
        if self._stream:
            DA.stream_synchronize(self._stream)
            DA.destroy_stream(self._stream)
        self._stream = 0
        self.slots = []
        self._arena = None
        self.residency.clear()


class NBXHeldExperts:
    """The routed experts of whole blocks held on ANOTHER card (S3, Prism's `ComponentAllocation.expert_cards`): every
    expert of each held layer copied once, at the store's build, into one arena on that card — each projection a
    256-byte-aligned view, the store's own layout — so the layer's fused op runs there against resident weights (its
    hidden state crosses to the card and back, `moe_fused_dispatch`). `layers` maps a layer key to its host expert
    triples [(gate, up, down)] in expert order."""

    def __init__(self, device_idx: int, layers):
        self.device_idx = int(device_idx)
        self.layers = {}
        shapes = {}
        total = 0
        for key, triples in layers.items():
            for trip in triples:
                for t in trip:
                    if not isinstance(t, NBXTensor) or t._device != "cpu" or not t.is_contiguous():
                        raise RuntimeError(
                            f"ZERO FALLBACK: an expert of {key!r} held on card {self.device_idx} is "
                            f"{type(t).__name__} on {getattr(t, '_device', '?')}; the card is filled from contiguous "
                            f"host experts")
                    total += _aligned(t._nbytes)
            shapes[key] = triples
        DA.set_device(self.device_idx)
        # one allocation per card: it lives as long as the component's store
        self._arena = NBXTensor.empty((total,), dtype=NBXDtype.uint8, device=f"cuda:{self.device_idx}")
        off = self._arena.data_ptr()
        for key, triples in shapes.items():
            g, u, d = [], [], []
            for trip in triples:
                views = []
                for t in trip:
                    v = NBXTensor.from_raw(off, tuple(t.shape), t._dtype, "cuda", owns_data=False,
                                           device_idx=self.device_idx, base=self._arena)
                    DA.memcpy(v.data_ptr(), t.data_ptr(), t._nbytes, 1)
                    views.append(v)
                    off += _aligned(t._nbytes)
                g.append(views[0]); u.append(views[1]); d.append(views[2])
            self.layers[key] = (g, u, d)
        DA.sync_device()
        self.bytes = total

    def release(self) -> None:
        self.layers = {}
        self._arena = None
