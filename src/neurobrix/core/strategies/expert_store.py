"""The `expert_store` component strategy: zero3's ratchet with the dense part resident and the routed
experts served from an expert cache on the card.

Prism selects it (`_zero3_resident_budgets`) for a routed component when its dense part plus one
layer's full expert set fit the room zero3 would have streamed through, but not every weight. The
plan carries two figures: `resident_weight_mb` (the dense part) and `expert_cache_mb` (the cache).

The runtime half is a different ratchet STATE, not a different class. `Zero3Strategy` builds it
when the component's allocation names `expert_store`, so every path that reaches zero3 reaches
it: plan-level zero3, and the lazy_sequential brick in both engines. The state holds:
* one resident pseudo-block -1 with every weight the ratchet would have streamed that is not a
  routed expert. It is bound before the priming sweep and re-bound at each pass's first op, like
  zero3's non-block weights, and no block streams;
* the expert weights, which stay on pinned host memory, their arena slots bound to the host tensors;
* one expert store per engine (torch: `core/runtime/graph/expert_store.py`, NBXTensor:
  `triton/expert_store.py`) sized `expert_cache_mb // one expert's bytes` slots. It is published
  as `_expert_store` on the sequence (compiled and triton MoE closures) and on the executor
  (triton-sequential dispatch), which call it with the experts each fused op routes to.

This module imports no tensor library: the engine's store is imported on the branch that runs it.
"""
from __future__ import annotations

from typing import Any, Dict, Set

from neurobrix.core.runtime.expert_residency import FUSED_OP, ExpertResidency, routed_layers

#: The component strategy name, as Prism writes it into the plan.
EXPERT_STORE = "expert_store"

_MIB = 1024 * 1024


def expert_cache_mb(allocations: Dict[str, Any], component_name: str) -> float:
    alloc = allocations.get(component_name)
    return float(alloc.get('expert_cache_mb', 0.0)) if isinstance(alloc, dict) else 0.0


def build_expert_store_state(z3: Any, component_name: str, executor: Any, seq: Any, is_triton: bool,
                             blocks: Dict[int, Dict[str, Any]], dev_idx: int, transfer_stream: int,
                             transfer_stream_torch: Any) -> Dict[str, Any]:
    """The ratchet state of an `expert_store` component (see the module docstring). `z3` is the
    Zero3Strategy whose helpers (snapshot, byte count, prefetch, install) the state reuses."""
    layers = routed_layers(getattr(executor, '_dag', None) or {})
    if not layers:
        raise RuntimeError(
            f"ZERO FALLBACK: {component_name} is planned `expert_store` but its executed graph holds "
            f"no fused routed layer (custom::moe_fused) — the plan and the graph disagree")
    expert_tids: Set[str] = {t for ids in layers.values() for x in ids for t in x}
    dense = [t for t in dict.fromkeys(t for e in blocks.values() for t in e.get('weight_tensor_ids', []))
             if t not in expert_tids]
    dense_entry = {'first_op': 0, 'last_op': -1, 'weight_tensor_ids': dense}
    cpu_originals = z3._snapshot_cpu_originals(executor, {-1: dense_entry})
    dense_bytes = z3._host_bytes(dense_entry, cpu_originals)
    resident_mb = z3._resident_budget_mb(component_name)
    if dense_bytes > resident_mb * _MIB:
        raise RuntimeError(
            f"ZERO FALLBACK: {component_name}'s dense part is {dense_bytes / _MIB:.1f} MB on the host and "
            f"the plan keeps {resident_mb:.0f} MB resident — Prism priced a different dense part")

    # One expert's (gate, up, down) host triple sets the slot geometry; every layer must share it.
    layer0 = next(iter(layers))
    geometry = None
    like = None
    for key, (g, u, d) in layers.items():
        for e in range(len(g)):
            trip = z3._snapshot_cpu_originals(executor, {0: {'weight_tensor_ids': [g[e], u[e], d[e]]}})
            if len(trip) != 3:
                raise RuntimeError(
                    f"ZERO FALLBACK: expert {e} of {key!r} has {len(trip)} of its 3 weights in the "
                    f"executor's weights — the store copies host experts")
            t3 = (trip[g[e]], trip[u[e]], trip[d[e]])
            geo = tuple((tuple(t.shape), str(getattr(t, 'dtype', None) if not hasattr(t, '_dtype')
                                             else t._dtype)) for t in t3)
            if geometry is None:
                geometry, like = geo, t3
            elif geo != geometry:
                raise RuntimeError(
                    f"ZERO FALLBACK: expert {e} of {key!r} is {geo}, expert 0 of {layer0!r} is {geometry} "
                    f"— the store's slots hold one expert geometry per component")
    slot_bytes = sum(z3._host_bytes({'weight_tensor_ids': ['x']}, {'x': t}) for t in like)
    cache_mb = expert_cache_mb(z3.context.allocations, component_name)
    n_slots = int(cache_mb * _MIB // slot_bytes) if slot_bytes else 0
    widest = max(len(ids[0]) for ids in layers.values())
    ops = (getattr(executor, '_dag', None) or {}).get('ops') or {}
    top_k = max(int((op.get('attributes') or {}).get('top_k', 0)) for op in ops.values()
                if op.get('op_type') == FUSED_OP)
    if n_slots < top_k:
        raise RuntimeError(
            f"ZERO FALLBACK: {component_name}'s expert cache is {cache_mb:.0f} MB = {n_slots} slots of "
            f"{slot_bytes / _MIB:.2f} MB, below the top_k={top_k} experts one token reads — Prism "
            f"selects expert_store only when the routed working set fits")

    residency = ExpertResidency(n_slots)
    if is_triton:
        from neurobrix.triton.expert_store import NBXExpertStore
        store = NBXExpertStore(residency, dev_idx, like)
    else:
        import torch
        from neurobrix.core.runtime.graph.expert_store import TorchExpertStore
        store = TorchExpertStore(residency, torch.device(f"cuda:{dev_idx}"), like)
    seq._expert_store = store
    executor._expert_store = store

    state: Dict[str, Any] = {
        'seq': seq,
        'is_triton': is_triton,
        'exec_dev_idx': dev_idx,
        'blocks': {-1: dense_entry},
        'op_to_block': {},
        'cpu_originals': cpu_originals,
        'gpu_cache': {},
        'block_events': {},
        'current_block': -1,
        'transfer_stream': transfer_stream,
        'transfer_stream_torch': transfer_stream_torch,
        'primed': False,
        'executor': executor,
        'resident': {-1},
        'streamed': [],
        'fresh_pass': False,
        'expert_store': store,
    }
    z3._ratchet[component_name] = state
    z3._prefetch_block(state, -1, is_triton)
    z3._wait_for_block(state, -1, is_triton)
    z3._install_block_on_arena(state, -1)
    print(
        f"[ExpertStore] {component_name}: dense {dense_bytes / _MIB:.0f} MB resident (plan "
        f"{resident_mb:.0f} MB), {len(layers)} routed layers x {widest} experts on the host, cache "
        f"{n_slots} slots x {slot_bytes / _MIB:.2f} MB = {n_slots * slot_bytes / _MIB:.0f} MB (plan "
        f"{cache_mb:.0f} MB), path={'triton' if is_triton else 'native'}", flush=True)
    return state


def release_expert_store(state: Dict[str, Any]) -> None:
    store = state.get('expert_store')
    if store is None:
        return
    store.release()
    for holder in (state.get('seq'), state.get('executor')):
        if holder is not None and getattr(holder, '_expert_store', None) is store:
            holder._expert_store = None
