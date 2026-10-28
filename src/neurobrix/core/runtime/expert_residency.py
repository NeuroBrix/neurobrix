"""Which routed experts an expert store keeps on the card — the policy, shared by both engines.

The `expert_store` component strategy (Prism, `_zero3_resident_budgets`) keeps a routed
component's dense weights resident and its experts on pinned host memory, with an expert cache
of `expert_cache_mb` on the card. This module is the part of that cache the two engines share:
WHICH expert sits in WHICH device slot. It moves no byte and imports no tensor library (R33):
the compiled store (`core/runtime/graph/expert_store.py`, torch) and the triton store
(`triton/expert_store.py`, NBXTensor) each own their slots and their copies.

The policy is one least-recently-used set over the whole component: a slot is keyed by
(layer, expert), never partitioned per layer, so a layer whose routing is concentrated lends its
room to one whose routing is spread. Every expert a call routes to is pinned for that call — the
call's own misses never evict one another, and a call that routes to more distinct experts than
there are slots is refused by name (`assign`), never served by a partial set.

Prior art: Eliseev & Mazur, "Fast Inference of Mixture-of-Experts Language Models with
Offloading" (arXiv:2312.17238) — an LRU expert cache per layer on the card, with a speculative
prefetch of the next layer's experts; MoE-Infinity (arXiv:2401.14361) — an activation-aware cache.
Ours is global LRU, demand-fetched, plus the PREFETCH of step 3 (2026-10-18): while layer L's
experts compute, the experts the PREVIOUS token routed to at layer L+1 are copied ahead (temporal
locality of MoE routing: the prediction is the last routing seen at that layer, nothing learned
elsewhere), into slots the current call does not hold (`protect`); the real call at L+1 finds them
resident and waits only for their copies' events. The layer order is the order the calls arrive in
(`note`), never a name; a prefetch is planned only when the next layer's predicted set and the
current call both fit in the slots (`plan_prefetch`), else none, the reason kept for the summary.
`NBX_EXPERT_PREFETCH=0` keeps the demand-fetched path — the oracle a prefetching run is judged
against (byte identity: the same bytes land in the slots, the experts compute in the same order).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Hashable, Iterable, List, Optional, Tuple

#: The fused op the runtime's MoE recogniser emits (`moe_fusion.detect_and_fuse_moe`).
FUSED_OP = "custom::moe_fused"


class ExpertResidency:
    """One LRU of `n_slots` device slots keyed by (layer, expert)."""

    def __init__(self, n_slots: int):
        if int(n_slots) <= 0:
            raise RuntimeError(f"ZERO FALLBACK: an expert store needs at least one slot, got {n_slots}")
        self.n_slots = int(n_slots)
        self._lru: "OrderedDict[Tuple[Hashable, int], int]" = OrderedDict()   # key -> slot, oldest first
        self._free: List[int] = list(range(self.n_slots - 1, -1, -1))         # popped from the end: 0 first
        self.hits = 0
        self.misses = 0
        # step 3: the layer order as the calls arrive, the last routing per layer, the prefetch ledger
        self.order: List[Hashable] = []
        self.last_routed: Dict[Hashable, List[int]] = {}
        self._prefetched: set = set()
        self.prefetched = 0          # experts copied ahead
        self.prefetch_used = 0       # ... that the next call found resident
        self.prefetch_wasted = 0     # ... evicted or unrouted before any call used them
        self.prefetch_skipped: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._lru)

    def reset(self) -> None:
        """Every slot free again, the map empty: another residency (the card's, for decode tokens) filled the slots
        since this one's last call, so none of its entries says what a slot holds any more."""
        self._lru.clear()
        self._free = list(range(self.n_slots - 1, -1, -1))
        self._prefetched.clear()
        self.last_routed = {}

    def slot_of(self, layer: Hashable, expert: int) -> Optional[int]:
        return self._lru.get((layer, int(expert)))

    def assign(self, layer: Hashable, routed: Iterable[int]) -> Tuple[Dict[int, int], List[int]]:
        """The slot of every expert `routed` names for this call, and the experts that MISSED (their
        slot holds something else until the caller copies them in). `routed` is read in ascending
        order — the order the engines compute the experts in — so the slot each miss receives is
        deterministic for a given history. Records the layer order and the call's routing (the
        prediction for the next token at this layer)."""
        experts = sorted({int(e) for e in routed})
        if len(experts) > self.n_slots:
            raise RuntimeError(
                f"ZERO FALLBACK: layer {layer!r} routes to {len(experts)} distinct experts in one call and "
                f"the expert store holds {self.n_slots} slots — the dispatchers acquire a wider call in "
                f"runs of at most n_slots experts")
        self.note(layer)
        self.last_routed[layer] = experts
        slots, missed = self._place(layer, experts, protect=())
        for e in experts:
            if (layer, e) in self._prefetched:
                self._prefetched.discard((layer, e))
                if e not in missed:
                    self.prefetch_used += 1
        self.hits += len(experts) - len(missed)
        self.misses += len(missed)
        return slots, missed

    def _place(self, layer: Hashable, experts: List[int], protect) -> Tuple[Dict[int, int], List[int]]:
        """Slots for `experts` at `layer`: hits moved to the recent end, misses given a free slot or the
        least recent slot outside this set and outside `protect` (keys another call still reads)."""
        call = {(layer, e) for e in experts}
        keep = call | set(protect)
        slots: Dict[int, int] = {}
        missed: List[int] = []
        for e in experts:
            slot = self._lru.get((layer, e))
            if slot is not None:
                self._lru.move_to_end((layer, e))
                slots[e] = slot
        for e in experts:
            if e in slots:
                continue
            if self._free:
                slot = self._free.pop()
            else:
                victim = next((k for k in self._lru if k not in keep), None)
                if victim is None:
                    raise RuntimeError(f"ZERO FALLBACK: expert store found no evictable slot for {layer!r}")
                if victim in self._prefetched:
                    self._prefetched.discard(victim)
                    self.prefetch_wasted += 1
                slot = self._lru.pop(victim)
            self._lru[(layer, e)] = slot
            slots[e] = slot
            missed.append(e)
        return slots, missed

    # ── step 3: the prefetch ─────────────────────────────────────────────────────────────────────────
    def note(self, layer: Hashable) -> None:
        """The layer order is the order the calls arrive in; a layer seen before adds nothing."""
        if layer not in self.last_routed and layer not in self.order:
            self.order.append(layer)

    def next_layer(self, layer: Hashable) -> Optional[Hashable]:
        """The layer the calls visit after `layer` (None at the last one, or before the order is known)."""
        try:
            i = self.order.index(layer)
        except ValueError:
            return None
        return self.order[i + 1] if i + 1 < len(self.order) else None

    def predicted(self, layer: Hashable) -> List[int]:
        """The experts the last call at `layer` routed to — the prediction for the next call there."""
        return list(self.last_routed.get(layer) or [])

    def plan_prefetch(self, layer: Hashable, protect: Iterable[Tuple[Hashable, int]]) -> List[int]:
        """The experts to copy ahead for `layer`: the predicted ones not resident, when the prediction
        and the protected call both fit in the slots; [] otherwise, the reason counted."""
        protect = set(protect)
        predicted = self.predicted(layer)
        if not predicted:
            self.prefetch_skipped["no prediction yet"] = self.prefetch_skipped.get("no prediction yet", 0) + 1
            return []
        if len(predicted) + len(protect) > self.n_slots:
            self.prefetch_skipped["slots"] = self.prefetch_skipped.get("slots", 0) + 1
            return []
        return [e for e in predicted if self._lru.get((layer, e)) is None]

    def reserve(self, layer: Hashable, experts: Iterable[int], protect: Iterable[Tuple[Hashable, int]]
                ) -> Dict[int, int]:
        """Slots for a prefetch of `experts` at `layer` (all misses by construction of `plan_prefetch`),
        never evicting a `protect` key; counted as prefetched, not as misses."""
        experts = sorted({int(e) for e in experts})
        slots, missed = self._place(layer, experts, protect=set(protect))
        for e in missed:
            self._prefetched.add((layer, e))
        self.prefetched += len(missed)
        return {e: slots[e] for e in missed}

    def summary(self) -> str:
        skipped = ", ".join(f"{k} x{v}" for k, v in sorted(self.prefetch_skipped.items())) or "none"
        return (f"slots={self.n_slots} layers={len(self.order)} hits={self.hits} misses={self.misses} "
                f"prefetched={self.prefetched} used={self.prefetch_used} wasted={self.prefetch_wasted} "
                f"prefetch skipped: {skipped}")

    def clear(self) -> None:
        self._lru.clear()
        self._free = list(range(self.n_slots - 1, -1, -1))
        self._prefetched.clear()


def routed_layers(dag: Dict[str, Any]) -> Dict[str, Tuple[List[str], List[str], List[str]]]:
    """{layer key: (gate, up, down expert weight ids)} for every fused routed op of `dag` (the graph
    as executed, after the MoE fusion). The layer key is the first expert's gate weight id — the
    same string every engine's dispatch reads from the op's attributes, so the three dispatchers
    (compiled, triton, triton-sequential) name a layer identically without an op uid.

    A stacked expert slab (one [E, ...] parameter per projection) has no per-expert host tensor
    to copy into a slot; it is refused by name — Prism leaves such a component on zero3."""
    out: Dict[str, Tuple[List[str], List[str], List[str]]] = {}
    for uid in (dag.get("execution_order") or list((dag.get("ops") or {}).keys())):
        op = (dag.get("ops") or {}).get(uid) or {}
        if op.get("op_type") != FUSED_OP:
            continue
        a = op.get("attributes") or {}
        if a.get("stacked_experts"):
            raise RuntimeError(
                f"ZERO FALLBACK: fused MoE op {uid} stores its experts as a stacked slab — the expert "
                "store copies per-expert matrices and cannot serve it (Prism keeps it on zero3)")
        ids = tuple(list(a.get(f) or []) for f in
                    ("expert_gate_weight_ids", "expert_up_weight_ids", "expert_down_weight_ids"))
        n = int(a["num_experts"])
        if any(len(x) != n for x in ids):
            raise RuntimeError(f"ZERO FALLBACK: fused MoE op {uid} lists {[len(x) for x in ids]} expert "
                               f"weights for {n} experts")
        out[ids[0][0]] = ids
    return out
