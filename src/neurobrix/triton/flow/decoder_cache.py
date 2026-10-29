"""The KV cache a decoding flow opens on its decoder — ONE opener for every flow of the Triton engine.

Mirror of `core/flow/decoder_cache.py` (R30): the same graph reader (`core/flow/decoder_kv`, pure
dict walking), the same plan check (`lm_facts.decoder_cache_from_plan`), the NBXTensor paged cache
and interceptor of `triton/kv_cache.py`. No torch (R33). `NBX_KV_RECOMPUTE=1` keeps the recompute
oracle.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Optional

from nbxtensor.dtype.nbx_dtype import parse_dtype


def open_decoder_cache(ctx, dec_name: str, max_tokens: int):
    """Build (once per executor) and reset the decoder's NBXTensor KV cache, interceptors registered
    on its self-attentions and positional ops; None when the recompute oracle is requested, when the
    graph has no self-attention, or when nothing could place a decoded token at its position."""
    if os.environ.get("NBX_KV_RECOMPUTE") == "1":
        return None
    executor = ctx.executors.get(dec_name)
    dag = getattr(executor, "_dag", None) if executor is not None else None
    if not dag:
        return None
    from neurobrix.core.flow.decoder_kv import decoder_self_attention_plan, positions_are_offset_by
    plan = decoder_self_attention_plan(dag)
    if plan is None:
        return None
    if not positions_are_offset_by(plan):
        print(f"[{dec_name}] KV cache REFUSED: the decoder graph takes no position_ids and carries no "
              f"positional arange, table slice or count the cache could offset — recompute path "
              f"(D-STT-KV-WHISPER-LARGE)", file=sys.stderr, flush=True)
        return None
    from neurobrix.triton.kv_cache import TritonAttentionInterceptor
    interceptor = getattr(executor, "_decoder_kv_interceptor", None)
    if interceptor is None:
        dtype = parse_dtype(str(getattr(executor, "dtype", None) or "float16"))
        from neurobrix.core.runtime.lm_facts import decoder_cache_facts, decoder_cache_from_plan
        kv = decoder_cache_from_plan(getattr(ctx.plan, "kv_cache_plan", None),
                                     decoder_cache_facts(dag), max_tokens, dec_name)
        from neurobrix.triton.flow.autoregressive import paged_kv_cache
        cache = paged_kv_cache(dict(num_layers=kv.num_layers, num_kv_heads=kv.num_kv_heads,
                                    k_head_dim=kv.k_head_dim, v_head_dim=kv.v_head_dim,
                                    max_cache_len=int(kv.max_cache_len), dtype=dtype),
                               getattr(ctx.plan, "kv_cache_plan", None))
        interceptor = TritonAttentionInterceptor(cache=cache, num_heads=plan["num_heads"])
        variant = {
            "aten::_scaled_dot_product_efficient_attention": interceptor.intercept_efficient,
            "aten::_scaled_dot_product_cudnn_attention": interceptor.intercept_efficient,
            "aten::_scaled_dot_product_flash_attention": interceptor.intercept_flash,
        }
        per_uid = {uid: variant.get(dag["ops"][uid]["op_type"], interceptor.intercept)
                   for uid in plan["self_attn_uids"]}
        for uid in plan["arange_uids"]:
            per_uid[uid] = interceptor.intercept_arange
        for uid in plan["position_slice_uids"]:
            per_uid[uid] = interceptor.intercept_position_slice
        for uid in plan["position_cumsum_uids"]:
            per_uid[uid] = interceptor.intercept_position_cumsum
        executor.register_op_uid_interceptors(per_uid)
        executor._decoder_kv_interceptor = interceptor
        how = ("positions from the caller" if plan["positions_input"] else
               f"{len(plan['arange_uids'])} arange, {len(plan['position_slice_uids'])} table slice, "
               f"{len(plan['position_cumsum_uids'])} count offset")
        print(f"   [{dec_name}] KV cache (triton): {plan['num_layers']} self-attention layers cached, "
              f"{len(plan['cross_attn_uids'])} cross-attentions native, max={kv.max_cache_len}, {how}")
    interceptor.reset()
    return interceptor


class DecodeSteps:
    """The step protocol of a cached decode loop, for one sequence of the cache (a branch) — mirror of
    `core/flow/decoder_cache.DecodeSteps`: the whole context at the first run (the prefill), the one
    new token afterwards, placed at its absolute position by the cache; without a cache every run
    sees the whole context (the recompute oracle)."""

    def __init__(self, interceptor: Optional[Any], branch: Optional[Any] = None):
        self.wrapper = interceptor
        self.branch = branch
        self.runs = 0

    @property
    def cached(self) -> bool:
        return self.wrapper is not None

    @property
    def one_token(self) -> bool:
        return self.wrapper is not None and self.runs >= 1

    def feed(self, whole, last):
        return last if self.one_token else whole

    def before_run(self) -> None:
        if self.wrapper is None:
            return
        if self.branch is not None:
            self.wrapper.use_branch(self.branch)
        if self.runs >= 1:
            # EVERY decode step, the first included: the offset the positional ops of the graph are shifted by is
            # the cache's length now. The first form shifted from the second step on and relied on `set_decode_mode`
            # for the first — the compiled wrapper sets the offset there, the Triton interceptor did not, so
            # openaudio's first cached token was rotated at position 0 on the Triton engine (4 % off the oracle at
            # its first step, 2026-10-19 02:46; the compiled step was bit-identical).
            self.wrapper.update_position_offset()

    def after_run(self) -> None:
        self.runs += 1
        if self.wrapper is not None and self.runs == 1:
            self.wrapper.set_decode_mode()
