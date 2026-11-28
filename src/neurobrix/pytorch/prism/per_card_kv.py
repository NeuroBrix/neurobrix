"""Per-card KV cache accounting and block-sharded weight attribution for PrismSolver."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def fgp_per_card_weights_bytes(
    parse_blocks_fn, container, comp_name: str, mem, shard_map: Dict[str, str],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Per-card compute-dtype weight bytes and block counts for a block-sharded (`fgp:`) component."""
    if container is None or not shard_map:
        return {}, {}
    blocks = parse_blocks_fn(container, comp_name)
    block_sizes = blocks.get("block_sizes") or {}
    block_keys_map = blocks.get("blocks") or {}
    non_block_mb = float(blocks.get("non_block_mb") or 0.0)
    non_block_keys = blocks.get("non_block_keys") or []
    disk_mb = sum(block_sizes.values()) + non_block_mb
    if disk_mb <= 0:
        return {}, {}
    scale = float(mem.weight_bytes) / (disk_mb * 1024 * 1024)
    weights_by_card: Dict[str, float] = {}
    blocks_by_card: Dict[str, int] = {}
    for b_num, keys in block_keys_map.items():
        dev = next((shard_map.get(k) for k in keys if k in shard_map), None)
        if dev is not None:
            weights_by_card[dev] = weights_by_card.get(dev, 0.0) + float(block_sizes.get(b_num, 0.0)) * 1024 * 1024 * scale
            blocks_by_card[dev] = blocks_by_card.get(dev, 0) + 1
    _add_non_block_weights(weights_by_card, non_block_mb, non_block_keys, shard_map, scale)
    return {d: int(b) for d, b in weights_by_card.items()}, blocks_by_card


def _add_non_block_weights(
    weights_by_card: Dict[str, float], non_block_mb: float, non_block_keys: list,
    shard_map: Dict[str, str], scale: float,
) -> None:
    """Distribute non-block weight bytes proportionally across cards holding those keys."""
    if non_block_mb <= 0 or not non_block_keys:
        return
    nb_counts: Dict[str, int] = {}
    for k in non_block_keys:
        dev = shard_map.get(k)
        if dev is not None:
            nb_counts[dev] = nb_counts.get(dev, 0) + 1
    nb_total = sum(nb_counts.values())
    if nb_total <= 0:
        return
    nb_bytes = non_block_mb * 1024 * 1024 * scale
    for dev, cnt in nb_counts.items():
        weights_by_card[dev] = weights_by_card.get(dev, 0.0) + nb_bytes * (cnt / nb_total)


def _component_cost_on_card(
    name: str, m: Any, d_name: str, dev: Any, fgp_cache: Dict[str, Dict[str, int]],
    lm: str, kv_already_counted: int, is_unified_fn, profile,
) -> int:
    """Return held bytes of component `name` on card `d_name`."""
    if name in fgp_cache:
        cost = int(fgp_cache[name].get(d_name, 0) + m.activation_bytes + m.overhead_bytes)
    elif isinstance(dev, str) and dev.startswith("zero3:") and not is_unified_fn(dev, profile):
        cost = int(m.activation_bytes + m.overhead_bytes)
    else:
        cost = int(m.total_bytes)
    return max(cost - int(kv_already_counted), 0) if name == lm else cost


def _build_fgp_weight_cache(component_memory, unloaded, strat_allocs, container, fgp_weights_fn):
    """Precompute per-card weight bytes for every block-sharded component."""
    cache: Dict[str, Dict[str, int]] = {}
    for name, m in component_memory.items():
        if name in unloaded:
            continue
        alloc = strat_allocs.get(name)
        dev = alloc[0] if isinstance(alloc, tuple) else alloc
        smap = alloc[1] if isinstance(alloc, tuple) and len(alloc) >= 2 else None
        if isinstance(dev, str) and dev.startswith("fgp:") and smap:
            cache[name], _ = fgp_weights_fn(container, name, m, smap)
    return cache


def per_card_kv_accounting(
    *, lm: Optional[str], strat_name: str, strat_allocs: Dict, component_memory: Dict,
    cards, profile, kv_already_counted: int, container,
    fgp_weights_fn, unloaded: set, alloc_devices_fn,
    loads_on_demand: bool, phase_peak_fn, is_unified_fn,
) -> Optional[Tuple[Dict[str, int], Dict[str, float]]]:
    """Per-card (held_bytes, kv_layer_share) when a decoder's blocks are spread over several cards (`fgp:`)."""
    if not cards or len(cards) <= 1 or container is None or lm is None or lm not in strat_allocs:
        return None
    lm_alloc = strat_allocs[lm]
    lm_dev = lm_alloc[0] if isinstance(lm_alloc, tuple) else lm_alloc
    lm_smap = lm_alloc[1] if isinstance(lm_alloc, tuple) and len(lm_alloc) >= 2 else None
    if not (isinstance(lm_dev, str) and lm_dev.startswith("fgp:") and lm_smap):
        return None
    lm_mem = component_memory.get(lm)
    if lm_mem is None:
        return None
    _, lm_blocks = fgp_weights_fn(container, lm, lm_mem, lm_smap)
    total_blocks = sum(lm_blocks.values())
    if total_blocks <= 0:
        return None
    card_names = [d.device_string for d in cards]
    kv_share = {d: lm_blocks.get(d, 0) / total_blocks for d in card_names}
    fgp_cache = _build_fgp_weight_cache(component_memory, unloaded, strat_allocs, container, fgp_weights_fn)
    held_by_card: Dict[str, int] = {}
    for d_name in card_names:
        costs: Dict[str, int] = {}
        for name, m in component_memory.items():
            if name in unloaded:
                continue
            alloc = strat_allocs.get(name)
            if alloc is None or d_name not in alloc_devices_fn(alloc):
                continue
            dev = alloc[0] if isinstance(alloc, tuple) else alloc
            costs[name] = _component_cost_on_card(
                name, m, d_name, dev, fgp_cache, lm, kv_already_counted, is_unified_fn, profile,
            )
        held_by_card[d_name] = phase_peak_fn(container, costs) if loads_on_demand else sum(costs.values())
    return held_by_card, kv_share


def clamp_remaining_by_card_kv(
    remaining: int, kv_growth_reserve: int, kv_only, bounded, per_card_kv,
    kv_capacity_fn, kv_growth_fn,
) -> int:
    """Clamp total remaining KV room by the tightest per-card block-sharded KV headroom."""
    if per_card_kv is None:
        return remaining
    held_by_card, kv_share = per_card_kv
    caps = [
        int(max(kv_capacity_fn(d, bounded) - held_by_card[d.device_string] - kv_growth_fn([d], bounded), 0)
            / kv_share[d.device_string])
        for d in kv_only if kv_share.get(d.device_string, 0.0) > 0
    ]
    return min(remaining, min(caps) + kv_growth_reserve) if caps else remaining


def finalize_rung_kv_room(
    per_card_fn, kv_capacity_fn, kv_growth_fn,
    strat_name, strat_allocs, component_memory, kv_only, profile,
    kv_already_counted, container, remaining: int, kv_growth_reserve: int, bounded,
):
    """Compute optional per-card KV accounting and clamp remaining KV room for `_solve_at_rung`."""
    per_card_kv = (
        per_card_fn(strat_name, strat_allocs, component_memory, kv_only, profile, kv_already_counted, container=container)
        if kv_only and len(kv_only) > 1 else None
    )
    remaining = clamp_remaining_by_card_kv(remaining, kv_growth_reserve, kv_only, bounded, per_card_kv, kv_capacity_fn, kv_growth_fn)
    return per_card_kv, remaining


def kv_card_room_map(cards, held: int, kv_plan, per_card, mode: str, engine_of_fn, rung_part_mb_fn) -> Optional[Dict[str, int]]:
    """Compute per-card remaining room bytes after held components and KV cache on Triton."""
    if kv_plan is None or not cards or engine_of_fn(mode) != "triton":
        return None
    if len(cards) == 1:
        room = int(rung_part_mb_fn(cards[0]) * 1024 * 1024)
        return {cards[0].device_string: room - held - int(kv_plan.memory_bytes)}
    if per_card is not None:
        held_by_card, kv_share = per_card
        return {
            d.device_string: (
                int(rung_part_mb_fn(d) * 1024 * 1024)
                - int(held_by_card.get(d.device_string, 0))
                - int(kv_plan.memory_bytes * kv_share.get(d.device_string, 0.0))
            )
            for d in cards
        }
    return None


def compute_fgp_drain_costs_mb(fgp_weights_fn, container, name: str, m, shard_map, by_name, whole_mb: Dict[str, float]) -> None:
    """Accumulate per-card whole-component cost (MB) for a block-sharded component."""
    w_by_card, _ = fgp_weights_fn(container, name, m, shard_map)
    mib = 1024 * 1024
    for d_key, w_bytes in w_by_card.items():
        if d_key in by_name:
            cost_mb = (w_bytes + m.activation_bytes + m.overhead_bytes) / mib
            whole_mb[d_key] = max(whole_mb.get(d_key, 0.0), cost_mb)


def compute_kv_floor_bytes(defaults: dict, max_context: Optional[int], token_bytes: int) -> int:
    """Compute served KV floor bytes bounded by optional `--max-context`."""
    from neurobrix.core.state.price import kv_min_tokens, DEFAULT_PROMPT_MARGIN
    floor_tokens = kv_min_tokens(defaults.get("max_tokens") or 0, defaults.get("prompt_margin", DEFAULT_PROMPT_MARGIN), True)
    if max_context is not None:
        floor_tokens = min(floor_tokens, max_context)
    return floor_tokens * token_bytes


def serialize_plan_kv_dict(kv_plan) -> Optional[Dict[str, Any]]:
    """Serialize KVCachePlan fields into the plan JSON dict."""
    if kv_plan is None:
        return None
    return {
        "max_cache_len": kv_plan.max_cache_len,
        "num_layers": kv_plan.num_layers,
        "num_kv_heads": kv_plan.num_kv_heads,
        "k_head_dim": kv_plan.k_head_dim,
        "v_head_dim": kv_plan.v_head_dim,
        "head_dim": kv_plan.head_dim,
        "dtype": kv_plan.dtype,
        "memory_mb": kv_plan.memory_mb,
        "memory_bytes": kv_plan.memory_bytes,
        "per_token_bytes": kv_plan.per_token_bytes,
        "initial_cache_len": kv_plan.initial_cache_len,
        "session_reach": kv_plan.session_reach,
        "prefill_chunk": kv_plan.prefill_chunk,
        "prefill_chunk_reason": kv_plan.prefill_chunk_reason,
        "max_context": kv_plan.max_context,
    }


def load_kv_cache_plan_dict(kv_data: Optional[Dict[str, Any]], kv_cls):
    """Reconstruct a KVCachePlan instance from serialized plan JSON."""
    if not kv_data:
        return None
    return kv_cls(
        max_cache_len=kv_data["max_cache_len"],
        num_layers=kv_data["num_layers"],
        num_kv_heads=kv_data["num_kv_heads"],
        k_head_dim=kv_data.get("k_head_dim") or kv_data["head_dim"],
        v_head_dim=kv_data.get("v_head_dim") or kv_data["head_dim"],
        dtype=kv_data["dtype"],
        memory_bytes=int(kv_data["memory_bytes"]),
        per_token_bytes=kv_data.get("per_token_bytes", 0),
        initial_cache_len=int(kv_data["initial_cache_len"]),
        session_reach=kv_data.get("session_reach"),
        prefill_chunk=kv_data.get("prefill_chunk"),
        prefill_chunk_reason=kv_data.get("prefill_chunk_reason", ""),
        max_context=kv_data.get("max_context"),
    )


def extract_max_context(input_config) -> Optional[int]:
    """Return validated `--max-context` cap from `input_config`, or None when uncapped."""
    from neurobrix.core.state.price import validate_max_context
    return validate_max_context(getattr(input_config, "max_context", None))


def make_pages_policy_decision(memory_bytes: int):
    """Construct the standard pages PolicyDecision for a fitting KV cache."""
    from neurobrix.core.state.policy import PolicyDecision
    return PolicyDecision("pages", f"pages: the whole cache fits ({memory_bytes / 2**20:,.0f} MB)")
