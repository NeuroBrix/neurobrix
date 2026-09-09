"""Install the self-attention KV cache on a decoder — one installer, every flow.

A decoder is a decoder: what a KV cache may hold, and whether the decoder can be advanced one
token at a time, are properties of the GRAPH, not of the flow driving it.
`core.flow.decoder_kv.decoder_self_attention_plan` reads them from the dataflow (self-attention
against cross-attention, the positional mechanism, the geometry); this module turns that plan into
a registered interceptor, or refuses out loud and lets the caller keep its recompute path.

It exists because the encoder_decoder flow had this code and the audio_llm flow had none — so a
listening model re-ran its whole context at every token, which is a quadratic cost and the reason
those rows decode at one token per second. The fix is not a second copy of the code.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Optional


def install_self_attention_kv(executor: Any, *, max_tokens: int, label: str) -> Optional[Any]:
    """Return the executor's attention interceptor with its cache registered, or None.

    None with a printed reason when: the recompute oracle is asked for (`NBX_KV_RECOMPUTE=1`),
    the graph has no attention, or the graph carries no positional mechanism the decode step can
    use. The interceptor is memoised on the executor — the cache's buffers must outlive one
    request, since a frozen replay plan records their device addresses — and `reset()` is called
    on every install so a new request starts at position zero.
    """
    if os.environ.get("NBX_KV_RECOMPUTE") == "1":
        return None
    dag = getattr(executor, "_dag", None) if executor is not None else None
    if not dag:
        return None

    from neurobrix.core.flow.decoder_kv import decoder_self_attention_plan
    plan = decoder_self_attention_plan(dag)
    if plan is None:
        return None

    # Three ways a decode step can place the token it feeds: an internal arange the cache
    # offsets, a positional table sliced to the token length whose window it moves, or the
    # CALLER supplying the position as a graph input — where there is nothing to offset and the
    # loop is responsible for passing the right one. A graph with none of the three would decode
    # every token at position 0, so the cache refuses and the recompute path stays.
    if not (plan["arange_uids"] or plan.get("position_slice_uids")
            or plan.get("uses_absolute_position")):
        print(f"[{label}] KV cache REFUSED: the decoder graph carries no positional arange, no "
              f"positional-table slice and no position input the step could place a token with — "
              f"recompute path (D-STT-KV-WHISPER-LARGE)", file=sys.stderr, flush=True)
        return None

    from neurobrix.kernels.nbx_tensor import parse_dtype
    from neurobrix.triton.kv_cache import TritonKVCache, TritonAttentionInterceptor
    interceptor = getattr(executor, "_decoder_kv_interceptor", None)
    if interceptor is None:
        dtype = parse_dtype(str(getattr(executor, "dtype", None) or "float16"))
        cache = TritonKVCache(num_layers=plan["num_layers"], num_kv_heads=plan["num_heads"],
                              k_head_dim=plan["head_dim"], v_head_dim=plan["head_dim"],
                              max_cache_len=int(max_tokens), dtype=dtype)
        interceptor = TritonAttentionInterceptor(cache=cache, num_heads=plan["num_heads"])
        variant = {
            "aten::_scaled_dot_product_efficient_attention": interceptor.intercept_efficient,
            "aten::_scaled_dot_product_cudnn_attention": interceptor.intercept_efficient,
            "aten::_scaled_dot_product_flash_attention": interceptor.intercept_flash,
        }
        per_uid = {uid: variant.get(dag["ops"][uid]["op_type"], interceptor.intercept)
                   for uid in plan["self_attn_uids"]}
        # An arange is offset ONLY when the caller does not supply the position: a model that
        # takes position_ids drives its rotary embedding from that input, and shifting an
        # internal arange as well would place every token twice (R23 — the same rule the
        # autoregressive flow applies through `uses_absolute_position`).
        if not plan.get("uses_absolute_position"):
            for uid in plan["arange_uids"]:
                per_uid[uid] = interceptor.intercept_arange
            for uid in plan.get("position_slice_uids") or []:
                per_uid[uid] = interceptor.intercept_position_slice
        executor.register_op_uid_interceptors(per_uid)
        executor._decoder_kv_interceptor = interceptor
        placed = ("the caller's position input" if plan.get("uses_absolute_position")
                  else f"{len(plan['arange_uids'])} arange(s) offset by the cache")
        print(f"   [{label}] KV cache (triton): {plan['num_layers']} self-attention layers cached, "
              f"{len(plan['cross_attn_uids'])} cross-attentions native, positions from {placed}")
    interceptor.reset()
    return interceptor
