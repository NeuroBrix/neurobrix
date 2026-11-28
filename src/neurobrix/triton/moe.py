"""The Triton engine's fused mixture-of-experts call: the engine's switches and diagnostics over the library's op.

The op is nbxtensor's (`nbxtensor.ops.moe_fused.fused_moe`): the routing, the experts' pointer tables, the host experts
promoted for a call, the one-token SIMT bands, the grouped GEMMs, an expert store's slots and the layer split over
several cards. What stays here is the engine's own choice:

  * two kill switches of the int4 one-token band, read at every call: NBX_MOE_VEC=0 sends the token through the
    grouped GEMMs, NBX_MOEV_DSPLIT=0 takes the band's fused down combine instead of the split one;
  * what a diagnostic prints: NBX_MOE_DIAG ("1": every call's intermediate values; any other value: the calls whose
    key contains it), NBX_Z3_TRITON_DIAG=1 (the live device memory around a host promotion);
  * what the engine's expert store does with a call's routing (`triton/expert_store.py`: the routing noted, the next
    layer's experts copied ahead).
"""
import logging
import os

import numpy as np

from nbxtensor.inspection import host_values
from nbxtensor.ops.moe_fused import fused_moe

_log = logging.getLogger(__name__)

# NBX_MOE_DIAG gate, hoisted to import time (C1 hygiene: never read the environ on the per-MoE-op hot path).
# Empty/unset: no dump at all.
_MOE_DIAG_ENV = os.environ.get("NBX_MOE_DIAG", "")
# The two kill switches of the one-token int4 band and the memory log, read at import too (the Mac's reread of lot 3):
# NBX_MOE_VEC=0 refuses the band, NBX_MOEV_DSPLIT=0 its split down projection, NBX_Z3_TRITON_DIAG=1 logs the memory.
_INT4_BAND, _SPLIT_DOWN = (os.environ.get(name, "1") != "0" for name in ("NBX_MOE_VEC", "NBX_MOEV_DSPLIT"))
_LOG_MEMORY = os.environ.get("NBX_Z3_TRITON_DIAG") == "1"


def _dump_for(cache_key: str):
    """The call's value dump for the op, None when NBX_MOE_DIAG is unset; a value other than "1" keeps only the calls
    whose key contains it (the NBX_DUMP_TIDS_FILTER idiom)."""
    if not _MOE_DIAG_ENV:
        return None

    def dump(label, tensor):
        if _MOE_DIAG_ENV != "1" and _MOE_DIAG_ENV not in cache_key:
            return
        try:
            vals = host_values(tensor).astype(np.float64)
            _log.debug(f"[MOE_DIAG] {label:20s} shape={list(tensor.shape)} dtype={tensor.dtype} "
                       f"norm={float(np.sqrt(np.dot(vals, vals))):.4f} head10={[round(float(v), 4) for v in vals[:10]]}")
        except Exception as _e:
            _log.debug(f"[MOE_DIAG] {label} failed: {_e}")
    return dump




def _choice_route_numpy(
    logits_np: np.ndarray,
    top_k: int,
    norm_topk_prob: bool,
    choice_routing: dict,
    bias_np=None,
):
    """Pure NumPy fp32 choice-score routing (CPU path, ZERO Torch)."""
    logits = np.asarray(logits_np, dtype=np.float32)
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    T, E = logits.shape

    score_func = choice_routing.get("score_func")
    if score_func == "sigmoid":
        scores = (np.float32(1.0) / (np.float32(1.0) + np.exp(-logits))).astype(np.float32)
    elif score_func == "softmax":
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        ex = np.exp(shifted).astype(np.float32)
        scores = (ex / np.sum(ex, axis=-1, keepdims=True)).astype(np.float32)
    else:
        raise RuntimeError(
            f"ZERO FALLBACK: unsupported choice_routing score_func={score_func!r}"
        )

    if bias_np is not None:
        bias = np.asarray(bias_np, dtype=np.float32)
        choice_scores = (scores + bias).astype(np.float32)
    else:
        choice_scores = scores

    n_group = int(choice_routing.get("n_group", 1))
    topk_group = int(choice_routing.get("topk_group", 1))
    group_topk_k = int(choice_routing.get("group_topk_k", 1))
    mask_fill_value = float(choice_routing.get("mask_fill_value", 0.0))

    if n_group > 1 and topk_group < n_group:
        if E % n_group != 0:
            raise RuntimeError(
                f"ZERO FALLBACK: choice_routing n_group={n_group} does not divide E={E}"
            )
        group_size = E // n_group
        grouped = choice_scores.reshape(T, n_group, group_size)
        part = np.sort(grouped, axis=-1)[:, :, -group_topk_k:]
        group_scores = np.sum(part, axis=-1, dtype=np.float32)
        group_idx = np.argsort(-group_scores, axis=-1, kind="stable")[:, :topk_group]
        group_mask = np.zeros((T, n_group), dtype=bool)
        np.put_along_axis(group_mask, group_idx, True, axis=1)
        score_mask = np.broadcast_to(group_mask[:, :, None], (T, n_group, group_size)).reshape(T, E)
        choice_scores = np.where(score_mask, choice_scores, np.float32(mask_fill_value))

    indices = np.argsort(-choice_scores, axis=-1, kind="stable")[:, :top_k].astype(np.int64)
    weights = np.take_along_axis(scores, indices, axis=1).astype(np.float32)

    if norm_topk_prob:
        denom = np.sum(weights, axis=-1, keepdims=True, dtype=np.float32)
        norm_eps = float(choice_routing.get("norm_eps", 0.0))
        if norm_eps != 0.0:
            denom = denom + np.float32(norm_eps)
        weights = (weights / denom).astype(np.float32)

    scale = float(choice_routing.get("routed_scaling_factor", 1.0))
    if scale != 1.0:
        weights = (weights * np.float32(scale)).astype(np.float32)

    return weights, indices


def _choice_route_cpu_tensor(gate_scores, top_k, norm_topk_prob, choice_routing, choice_bias):
    """Run choice-score routing on a CPU NBXTensor via NumPy and wrap results back as CPU NBXTensors."""
    from nbxtensor.tensor import NBXDtype, NBXTensor

    logits_np = host_values(gate_scores).reshape(gate_scores.shape)
    bias_np = None if choice_bias is None else host_values(choice_bias).reshape(choice_bias.shape)
    w_np, i_np = _choice_route_numpy(logits_np, top_k, norm_topk_prob, choice_routing, bias_np)
    w_t = NBXTensor.empty_cpu(w_np.shape, dtype=NBXDtype.float32)
    w_t._base[...] = w_np
    i_t = NBXTensor.empty_cpu(i_np.shape, dtype=NBXDtype.int64)
    i_t._base[...] = i_np
    return w_t, i_t


def _device_choice_scores(logits, choice_routing: dict, choice_bias, dev_idx: int):
    """Compute unbiased routing scores and bias-adjusted choice scores on device."""
    from nbxtensor.tensor import NBXDtype
    from nbxtensor.ops import attention_wrappers as _aw
    from nbxtensor.ops import elementwise_wrappers as _ew
    from nbxtensor.memory.device_transfer import transfer_tensor as _xfer

    score_func = choice_routing.get("score_func")
    if score_func == "sigmoid":
        scores = _ew.sigmoid_wrapper(logits)
    elif score_func == "softmax":
        scores = _aw.softmax(logits, dim=-1)
    else:
        raise RuntimeError(f"ZERO FALLBACK: unsupported choice_routing score_func={score_func!r}")
    if choice_bias is None:
        return scores, scores
    bias = choice_bias
    if getattr(bias, "_device", "cuda") == "cpu":
        bias = bias.to_cuda(dev_idx)
    elif getattr(bias, "device_idx", dev_idx) != dev_idx:
        bias = _xfer(bias, dev_idx)
    return scores, _ew.add(scores, bias.to(NBXDtype.float32))


def choice_route_triton(
    gate_scores,
    top_k: int,
    norm_topk_prob: bool,
    choice_routing: dict,
    choice_bias=None,
):
    """Choice-score MoE routing in fp32 (Triton/nbxtensor twin of `choice_route_torch`, ZERO Torch).

    Runs on-device via `nbxtensor.ops` when `gate_scores` is a device `NBXTensor`,
    or in pure NumPy fp32 when `gate_scores` is a host/CPU `NBXTensor` or `np.ndarray`.
    Returns `(weights [T, top_k] fp32, indices [T, top_k] int64)`.
    """
    if isinstance(gate_scores, np.ndarray):
        return _choice_route_numpy(gate_scores, top_k, norm_topk_prob, choice_routing, choice_bias)

    from nbxtensor.tensor import NBXDtype, NBXTensor

    if getattr(gate_scores, "_device", "cuda") == "cpu":
        return _choice_route_cpu_tensor(gate_scores, top_k, norm_topk_prob, choice_routing, choice_bias)

    from nbxtensor.ops import attention_wrappers as _aw
    from nbxtensor.ops import elementwise_wrappers as _ew
    from nbxtensor.ops import index_wrappers as _iw
    from nbxtensor.ops import reduction_wrappers as _rw
    from nbxtensor.memory.device_transfer import transfer_tensor as _xfer

    logits = gate_scores.to(NBXDtype.float32)
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    T, E = logits.shape
    dev_idx = logits.device_idx
    dev_kind = getattr(logits, "_device", "cuda")
    dev_str = f"cuda:{dev_idx}" if dev_kind == "cuda" else dev_kind

    scores, choice_scores = _device_choice_scores(logits, choice_routing, choice_bias, dev_idx)

    n_group = int(choice_routing.get("n_group", 1))
    topk_group = int(choice_routing.get("topk_group", 1))
    group_topk_k = int(choice_routing.get("group_topk_k", 1))
    mask_fill_value = float(choice_routing.get("mask_fill_value", 0.0))

    if n_group > 1 and topk_group < n_group:
        if E % n_group != 0:
            raise RuntimeError(
                f"ZERO FALLBACK: choice_routing n_group={n_group} does not divide E={E}"
            )
        group_size = E // n_group
        grouped = choice_scores.reshape(T, n_group, group_size)
        group_top_vals, _ = _ew.topk_wrapper(grouped, group_topk_k, dim=-1)
        group_scores = _rw.sum_wrapper(group_top_vals, dim=-1)
        _, group_idx = _ew.topk_wrapper(group_scores, topk_group, dim=-1, largest=True, sorted=False)
        ones = NBXTensor.ones(group_idx.shape, dtype=NBXDtype.float32, device=dev_str)
        group_mask = _iw.scatter_wrapper(NBXTensor.zeros_like(group_scores), 1, group_idx, ones)
        score_mask = group_mask.unsqueeze(-1).expand(T, n_group, group_size).contiguous().reshape(T, E)
        inv_mask = _ew.bitwise_not_wrapper(score_mask.to(NBXDtype.bool_))
        choice_scores = _iw.masked_fill(choice_scores, inv_mask, mask_fill_value)

    _, indices = _ew.topk_wrapper(choice_scores, top_k, dim=-1, largest=True, sorted=False)
    if indices.nbx_dtype != NBXDtype.int64:
        indices = indices.to(NBXDtype.int64)
    weights = _iw.gather_wrapper(scores, 1, indices)

    if norm_topk_prob:
        denom = _rw.sum_wrapper(weights, dim=-1, keepdim=True)
        norm_eps = float(choice_routing.get("norm_eps", 0.0))
        if norm_eps != 0.0:
            denom = _ew.add(denom, norm_eps)
        weights = _ew.div(weights, denom)

    scale = float(choice_routing.get("routed_scaling_factor", 1.0))
    if scale != 1.0:
        weights = _ew.mul(weights, scale)

    return weights, indices


def execute_moe_fused(
    gate_scores,
    hidden_states,
    gate_weights, up_weights, down_weights,
    top_k: int,
    num_experts: int,
    norm_topk_prob: bool = True,
    cache_key: str = "",
    topk_indices=None,
    topk_weights=None,
    weights_per_call: bool = False,
    expert_store=None,
    expert_layer=None,
    choice_routing=None,
    choice_bias=None,
):
    """One fused MoE op of the Triton engine (`nbxtensor.ops.moe_fused.fused_moe`, whose arguments these are), with
    the engine's switches read and its diagnostics attached. `cache_key` names the call in the diagnostics;
    `expert_store` is the component's `NBXExpertStore`, `expert_layer` the layer it serves (the first gate weight id);
    `topk_indices` / `topk_weights` the multi-gate routing the graph blended (mirror of the compiled engine, R30);
    `choice_routing` / `choice_bias` a choice-score routing cluster (sigmoid/softmax + bias + group mask + scale)."""
    if choice_routing is not None:
        topk_weights, topk_indices = choice_route_triton(
            gate_scores, top_k=top_k, norm_topk_prob=norm_topk_prob,
            choice_routing=choice_routing, choice_bias=choice_bias,
        )
        return fused_moe(
            None, hidden_states, gate_weights, up_weights, down_weights, top_k, num_experts,
            norm_topk_prob=False, topk_indices=topk_indices, topk_weights=topk_weights,
            weights_per_call=weights_per_call, store=expert_store, layer=expert_layer,
            int4_band=_INT4_BAND, split_down=_SPLIT_DOWN,
            dump=_dump_for(cache_key), label=cache_key, log_memory=_LOG_MEMORY)
    return fused_moe(
        gate_scores, hidden_states, gate_weights, up_weights, down_weights, top_k, num_experts,
        norm_topk_prob=norm_topk_prob, topk_indices=topk_indices, topk_weights=topk_weights,
        weights_per_call=weights_per_call, store=expert_store, layer=expert_layer,
        int4_band=_INT4_BAND, split_down=_SPLIT_DOWN,
        dump=_dump_for(cache_key), label=cache_key, log_memory=_LOG_MEMORY)
