"""Choice-score MoE routing (PyTorch) and graph-level fusion pipeline helpers."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def choice_route_torch(
    gate_scores,
    top_k: int,
    norm_topk_prob: bool,
    choice_routing: Dict[str, Any],
    choice_bias=None,
):
    """Choice-score MoE routing in fp32 (PyTorch twin of `triton/moe.choice_route_triton`)."""
    import torch
    import torch.nn.functional as F
    from neurobrix.core.dtype.engine import routing_upcast_fp32

    logits = routing_upcast_fp32(gate_scores)
    if logits.dim() == 3:
        logits = logits.reshape(-1, logits.size(-1))
    T, E = logits.shape

    score_func = choice_routing.get("score_func")
    if score_func == "sigmoid":
        scores = torch.sigmoid(logits)
    elif score_func == "softmax":
        scores = F.softmax(logits, dim=-1)
    else:
        raise RuntimeError(f"ZERO FALLBACK: unsupported choice_routing score_func={score_func!r}")

    choice_scores = _apply_choice_bias_torch(scores, choice_bias)
    choice_scores = _apply_group_mask_torch(choice_scores, T, E, choice_routing)

    _, indices = torch.topk(choice_scores, k=top_k, dim=-1, largest=True, sorted=False)
    weights = scores.gather(1, indices)
    return _normalize_and_scale_torch(weights, norm_topk_prob, choice_routing), indices


def _apply_choice_bias_torch(scores, choice_bias):
    """Add optional fp32 choice bias to unbiased routing scores."""
    from neurobrix.core.dtype.engine import routing_upcast_fp32

    if choice_bias is None:
        return scores
    bias = routing_upcast_fp32(choice_bias)
    if bias.device != scores.device:
        bias = bias.to(scores.device)
    return scores + bias


def _apply_group_mask_torch(choice_scores, T: int, E: int, choice_routing: Dict[str, Any]):
    """Apply optional top-k group masking to choice scores."""
    import torch

    n_group = int(choice_routing.get("n_group", 1))
    topk_group = int(choice_routing.get("topk_group", 1))
    if not (n_group > 1 and topk_group < n_group):
        return choice_scores
    if E % n_group != 0:
        raise RuntimeError(f"ZERO FALLBACK: choice_routing n_group={n_group} does not divide E={E}")
    group_topk_k = int(choice_routing.get("group_topk_k", 1))
    mask_fill_value = float(choice_routing.get("mask_fill_value", 0.0))
    group_size = E // n_group
    grouped = choice_scores.view(T, n_group, group_size)
    group_scores = grouped.topk(group_topk_k, dim=-1).values.sum(dim=-1)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, largest=True, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores).scatter(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(-1).expand(T, n_group, group_size).reshape(T, E)
    return choice_scores.masked_fill(~score_mask.bool(), mask_fill_value)


def _normalize_and_scale_torch(weights, norm_topk_prob: bool, choice_routing: Dict[str, Any]):
    """Optionally renormalise top-k weights and apply routed_scaling_factor."""
    if norm_topk_prob:
        denom = weights.sum(dim=-1, keepdim=True)
        norm_eps = float(choice_routing.get("norm_eps", 0.0))
        if norm_eps != 0.0:
            denom = denom + norm_eps
        weights = weights / denom
    scale = float(choice_routing.get("routed_scaling_factor", 1.0))
    return weights * scale if scale != 1.0 else weights


def route_choice_on_device(
    gate_scores, choice_bias, compute_dev, top_k: int, norm_topk_prob: bool,
    choice_routing: Dict[str, Any], to_card_fn, dump_fn=None,
) -> Tuple[Any, Any]:
    """Move gate_scores and optional choice_bias to `compute_dev` and run `choice_route_torch`."""
    if gate_scores.device != compute_dev:
        gate_scores = to_card_fn(gate_scores, compute_dev)
    if choice_bias is not None and choice_bias.device != compute_dev:
        choice_bias = to_card_fn(choice_bias, compute_dev)
    scores, indices = choice_route_torch(
        gate_scores, top_k, norm_topk_prob, choice_routing, choice_bias=choice_bias,
    )
    if dump_fn is not None:
        dump_fn("choice_scores", scores)
        dump_fn("choice_indices", indices)
    return scores, indices


def run_moe_and_gdr_fusion(dag: Dict[str, Any], family, norm_topk_prob: bool, moe_tokens) -> Dict[str, Any]:
    """Run MoE fusion followed by Gated Delta Rule fusion on a component DAG."""
    from .moe_fusion import detect_and_fuse_moe
    from .gated_delta_rule_fusion import detect_and_fuse_gated_delta_rule

    dag = detect_and_fuse_moe(dag, family, norm_topk_prob=norm_topk_prob, tokens=moe_tokens)
    return detect_and_fuse_gated_delta_rule(dag)


def resolve_choice_routing_slot(attrs: Dict[str, Any], tid_to_slot: Dict[str, int]):
    """Return `(choice_routing, choice_bias_slot)` from fused MoE op attributes."""
    choice_routing = attrs.get("choice_routing")
    tid = choice_routing.get("choice_bias_tid") if isinstance(choice_routing, dict) else None
    return choice_routing, (tid_to_slot[tid] if tid is not None else None)
