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
        # In place (supervisor 2026-10-20 01:20 (b)): on a unified device whose routed experts are views of the
        # container's files (Prism's `expert_files`, `holds_expert_files`), a slot is a POINTER to the expert
        # where it lies — its pages bound in place (`DeviceAllocator.bind_mapped`), nothing copied, no arena.
        self.in_place = bool(in_place)
        self._bound = set()      # (layer, expert) whose pages are bound
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
                for s in src:
                    DA.bind_mapped(s.data_ptr(), s._nbytes)
                self._bound.add((layer, e))
                self.residency.misses += 1
            else:
                self.residency.hits += 1
            views[e] = src
        filler = views[min(views)]
        triples = [views.get(e, filler) for e in range(int(n_experts))]
        return ([t[0] for t in triples], [t[1] for t in triples], [t[2] for t in triples])

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
