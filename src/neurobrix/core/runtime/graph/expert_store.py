"""The compiled engine's expert store: device slots for routed experts, filled on a side stream.

Mode 1 (torch). The triton engine has its own (`triton/expert_store.py`, NBXTensor); the two share
the residency POLICY (`core/runtime/expert_residency.py`, no tensor library) and no compute code.

The component's expert weights stay on pinned host memory; the store preallocates `n_slots`
(gate, up, down) device triples at the experts' own shape and dtype. A fused MoE call names the
experts its tokens route to (`acquire`); every miss is copied into its slot on a side stream, one
event per expert, after that stream has waited for the compute stream — a slot is rewritten only
once the work reading its former expert has run. The dispatch then waits for each expert's event
just before computing that expert, so the copy of expert e+1 overlaps the matmuls of expert e.

Byte identity: a slot is a byte copy of the host weight at the same shape and contiguous layout as
the `.to(device)` the dispatch performs without a store, and the experts are computed in the same
ascending order — the output is the zero3 output.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import torch

from neurobrix.core.runtime.expert_residency import ExpertResidency

Triple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class TorchExpertStore:
    def __init__(self, residency: ExpertResidency, device: torch.device, like: Triple):
        self.residency = residency
        self.device = torch.device(device)
        self._shape = tuple((tuple(t.shape), t.dtype) for t in like)
        self.slots: List[Triple] = [
            tuple(torch.empty(s, dtype=d, device=self.device) for s, d in self._shape)
            for _ in range(residency.n_slots)]
        self._stream = torch.cuda.Stream(device=self.device)
        self.bytes_in = 0

    @property
    def slot_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.slots[0]) if self.slots else 0

    def acquire(self, layer, routed, host: Callable[[int], Triple]
                ) -> Tuple[Dict[int, Triple], Dict[int, "torch.cuda.Event"]]:
        """({expert: device triple}, {missed expert: its copy's event}) for the experts `routed` names."""
        slots, missed = self.residency.assign(layer, routed)
        events: Dict[int, torch.cuda.Event] = {}
        if missed:
            self._stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self._stream):
                for e in missed:
                    dst = self.slots[slots[e]]
                    src = host(e)
                    for i, (d, s) in enumerate(zip(dst, src)):
                        if (tuple(s.shape), s.dtype) != self._shape[i] or not s.is_contiguous():
                            raise RuntimeError(
                                f"ZERO FALLBACK: expert {e} of {layer!r} projection {i} is "
                                f"{tuple(s.shape)} {s.dtype} contiguous={s.is_contiguous()}; the store's "
                                f"slots are {self._shape[i]} contiguous — one geometry per component")
                        d.copy_(s, non_blocking=True)
                        self.bytes_in += s.numel() * s.element_size()
                    ev = torch.cuda.Event()
                    ev.record(self._stream)
                    events[e] = ev
        return {e: self.slots[s] for e, s in slots.items()}, events

    def release(self) -> None:
        torch.cuda.current_stream(self.device).wait_stream(self._stream)
        self.slots = []
        self.residency.clear()
