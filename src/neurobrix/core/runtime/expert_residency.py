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
Offloading" (arXiv:2312.17238) — an LRU expert cache per layer on the card; MoE-Infinity
(arXiv:2401.14361) — an activation-aware cache. Ours is global LRU, demand-fetched; a predictive
prefetch is a later lever, not this one.
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

    def __len__(self) -> int:
        return len(self._lru)

    def slot_of(self, layer: Hashable, expert: int) -> Optional[int]:
        return self._lru.get((layer, int(expert)))

    def assign(self, layer: Hashable, routed: Iterable[int]) -> Tuple[Dict[int, int], List[int]]:
        """The slot of every expert `routed` names for this call, and the experts that MISSED (their
        slot holds something else until the caller copies them in). `routed` is read in ascending
        order — the order the engines compute the experts in — so the slot each miss receives is
        deterministic for a given history."""
        experts = sorted({int(e) for e in routed})
        if len(experts) > self.n_slots:
            raise RuntimeError(
                f"ZERO FALLBACK: layer {layer!r} routes to {len(experts)} distinct experts in one call and "
                f"the expert store holds {self.n_slots} slots — the dispatchers acquire a wider call in "
                f"runs of at most n_slots experts")
        call = {(layer, e) for e in experts}
        slots: Dict[int, int] = {}
        missed: List[int] = []
        for e in experts:
            key = (layer, e)
            slot = self._lru.get(key)
            if slot is not None:
                self._lru.move_to_end(key)
                slots[e] = slot
                self.hits += 1
        for e in experts:
            if e in slots:
                continue
            if self._free:
                slot = self._free.pop()
            else:
                victim = next((k for k in self._lru if k not in call), None)
                if victim is None:      # unreachable: len(experts) <= n_slots and every key of `call` is new or moved
                    raise RuntimeError(f"ZERO FALLBACK: expert store found no evictable slot for {layer!r}")
                slot = self._lru.pop(victim)
            self._lru[(layer, e)] = slot
            slots[e] = slot
            missed.append(e)
            self.misses += 1
        return slots, missed

    def clear(self) -> None:
        self._lru.clear()
        self._free = list(range(self.n_slots - 1, -1, -1))


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
