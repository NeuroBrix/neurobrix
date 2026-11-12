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
):
    """One fused MoE op of the Triton engine (`nbxtensor.ops.moe_fused.fused_moe`, whose arguments these are), with
    the engine's switches read and its diagnostics attached. `cache_key` names the call in the diagnostics;
    `expert_store` is the component's `NBXExpertStore`, `expert_layer` the layer it serves (the first gate weight id);
    `topk_indices` / `topk_weights` the multi-gate routing the graph blended (mirror of the compiled engine, R30)."""
    return fused_moe(
        gate_scores, hidden_states, gate_weights, up_weights, down_weights, top_k, num_experts,
        norm_topk_prob=norm_topk_prob, topk_indices=topk_indices, topk_weights=topk_weights,
        weights_per_call=weights_per_call, store=expert_store, layer=expert_layer,
        int4_band=os.environ.get("NBX_MOE_VEC", "1") != "0",
        split_down=os.environ.get("NBX_MOEV_DSPLIT", "1") != "0",
        dump=_dump_for(cache_key), label=cache_key,
        log_memory=os.environ.get("NBX_Z3_TRITON_DIAG") == "1")
