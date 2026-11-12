"""The triton engine's expert store: what the engine decides about the library's expert slots.

Mode 2 (NBXTensor, zero torch — R33). The compiled engine has its own (`core/runtime/graph/expert_store.py`); the two
share the residency POLICY (`core/runtime/expert_residency.py`, no tensor library) and no compute code.

The slots themselves — the device arena, its views, the host-to-card copies and their stream, the card's own slot map
and its clock, the kernels' launches — are nbxtensor's (`nbxtensor.memory.expert_slots.ExpertSlotStore`; the blocks
another card holds whole: `HeldExperts`). The engine keeps what is a decision: WHICH expert sits in WHICH slot (the
residency handed to the store), how many slots and on which card (Prism's plan, `core/strategies/expert_store.py`),
whether the next layer's predicted experts are copied ahead (`on_enqueued`, opt-in), and the recording of the
decode's routing for Prism's reuse curve (`trace`). The library's fused op (`nbxtensor.ops.moe_fused`) asks and tells
the store through `decodes_on_card`, `on_routed` and `on_enqueued`, which this store answers with those decisions.
"""
from __future__ import annotations

import logging
import os

from nbxtensor.memory.expert_slots import ExpertSlotStore


class NBXExpertStore(ExpertSlotStore):
    def __init__(self, residency, device_idx: int, like, in_place: bool = False):
        self.trace = None     # an ExpertTrace (core/runtime/routing_reuse.py) the store notes each decode call into
        self.hosts = {}       # layer -> host expert reader, registered by its call
        super().__init__(residency, device_idx, like, in_place=in_place)

    def _acquire(self, layer, routed, n_experts: int, host):
        if self.trace is not None:
            self.trace.note(layer, routed)      # the routing reuse curve (core/runtime/routing_reuse.py)
        return super()._acquire(layer, routed, n_experts, host)

    def decodes_on_card(self, layer) -> bool:
        return self.trace is None and super().decodes_on_card(layer)    # a recorded routing is read on the host

    def on_routed(self, layer, routed, host) -> None:
        self.hosts[layer] = host    # the next layer's prefetch reads it (step 3)

    # ── step 3: the prefetch of the next layer's predicted experts (mirror of the compiled store's `prefetch_next`) ──
    def on_enqueued(self, layer, routed) -> int:
        """The fused op's call at `layer` has its launches enqueued: copy ahead, on the store's stream, the
        experts the previous token routed to at the layer that follows `layer`, into slots the call at `layer`
        does not hold; 0 when nothing can be planned or unless NBX_EXPERT_PREFETCH=1 asks for it (opt-in since
        2026-10-19: 21 % right at decode on Ming, +23 % time on the Triton engine). The stream is blocking
        against the legacy stream, so copies issued after the launch run behind it and ahead of the next
        layer's."""
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

    def release(self) -> None:
        if self.trace is not None:
            self.trace.write(f"recorded by the triton expert store's decode ({self.residency.summary()})")
            self.trace = None
        if self.residency.hits or self.residency.misses:
            logging.getLogger(__name__).debug(f"   {self.summary()}")
        super().release()
