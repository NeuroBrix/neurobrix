"""The triton engine's expert store: device slots for routed experts, filled on a transfer stream.

Mode 2 (NBXTensor, zero torch — R33). The compiled engine has its own
(`core/runtime/graph/expert_store.py`); the two share the residency POLICY
(`core/runtime/expert_residency.py`, no tensor library) and no compute code.

The component's expert weights stay on pinned host memory; the store preallocates `n_slots`
(gate, up, down) device triples at the experts' own shape and dtype. `execute_moe_fused` reads the
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

from typing import Callable, List, Tuple

from neurobrix.core.runtime.expert_residency import ExpertResidency
from neurobrix.kernels.nbx_tensor import DeviceAllocator as DA, NBXTensor


class NBXExpertStore:
    def __init__(self, residency: ExpertResidency, device_idx: int, like):
        self.residency = residency
        self.device_idx = int(device_idx)
        for t in like:
            if not isinstance(t, NBXTensor):
                raise RuntimeError(
                    f"ZERO FALLBACK: the expert store copies dense NBXTensor experts; got {type(t).__name__} "
                    "(a quantized expert carries packed weights + scales — not this store's geometry)")
        self._shape = tuple((tuple(t.shape), t._dtype) for t in like)
        DA.set_device(self.device_idx)
        self.slots: List[Tuple[NBXTensor, NBXTensor, NBXTensor]] = [
            tuple(NBXTensor.empty(s, dtype=d, device=f"cuda:{self.device_idx}") for s, d in self._shape)
            for _ in range(residency.n_slots)]
        self._stream = DA.create_stream()
        self.bytes_in = 0

    @property
    def slot_bytes(self) -> int:
        return sum(t._nbytes for t in self.slots[0]) if self.slots else 0

    def acquire(self, layer, routed, n_experts: int, host: Callable[[int], Tuple]):
        """(gate, up, down) lists over all `n_experts`, device-resident, for a call routing to `routed`."""
        slots, missed = self.residency.assign(layer, routed)
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

    def release(self) -> None:
        DA.set_device(self.device_idx)
        DA.stream_synchronize(self._stream)
        DA.destroy_stream(self._stream)
        self._stream = 0
        self.slots = []
        self.residency.clear()
