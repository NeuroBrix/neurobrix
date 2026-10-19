"""The KV cache a decoding flow opens on its decoder — ONE opener for every flow of the compiled engine.

A decoder's self-attentions, the positional op the cache offsets and the cache's geometry are read from
the decoder's GRAPH (`core/flow/decoder_kv.decoder_self_attention_plan`); the cache's length and page
size come from the PLAN (Prism priced it, `lm_facts.decoder_cache_from_plan`); the interceptors are
registered by op id on the executor the flow runs. The encoder-decoder flow carried this opener alone
(whisper-class, 2026-09); the state engine's step 3 (2026-10-18) hands it to the VLM, audio-LLM,
TTS-LLM and dual-AR flows, which re-forwarded their whole context at every token.

`NBX_KV_RECOMPUTE=1` keeps the recompute path: the O(n) oracle every cached decode is judged against
(tokens identical over a bounded decode). Mirror: `triton/flow/decoder_cache.py` (R30).
"""
from __future__ import annotations

import os
import sys
from typing import Any, Optional


def open_decoder_cache(ctx, dec_name: str, max_tokens: int):
    """Build (once per executor) and reset the decoder's KV cache, interceptors registered on its
    self-attentions and positional ops; None when the recompute oracle is requested, when the graph
    has no self-attention, or when nothing could place a decoded token at its position."""
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
    from neurobrix.core.runtime.graph.kv_cache_wrapper import KVCacheAttentionWrapper, KVCacheConfig
    wrapper = getattr(executor, "_decoder_kv_wrapper", None)
    if wrapper is None:
        dtype = getattr(executor, "dtype", None) or "float16"
        from neurobrix.core.runtime.lm_facts import decoder_cache_facts, decoder_cache_from_plan
        kv = decoder_cache_from_plan(getattr(ctx.plan, "kv_cache_plan", None),
                                     decoder_cache_facts(dag), max_tokens, dec_name)
        from neurobrix.core.module.cache.factory import paged_params
        page_tokens, async_release = paged_params(
            kv.num_kv_heads, kv.k_head_dim, kv.v_head_dim, dtype,
            (getattr(executor, "vendor", None), getattr(executor, "arch", None)),
            getattr(ctx.plan, "kv_cache_plan", None))
        config = KVCacheConfig(
            num_layers=kv.num_layers, num_kv_heads=kv.num_kv_heads,
            k_head_dim=kv.k_head_dim, v_head_dim=kv.v_head_dim,
            max_cache_len=int(kv.max_cache_len), dtype=str(dtype),
            page_tokens=page_tokens, async_release=async_release)
        wrapper = KVCacheAttentionWrapper(config, num_heads=plan["num_heads"])
        by_type = wrapper.get_interceptors()
        per_uid = {}
        for uid in plan["self_attn_uids"]:
            fn = by_type.get(dag["ops"][uid]["op_type"])
            if fn is None:
                raise RuntimeError(f"ZERO FALLBACK: no KV interceptor for {dag['ops'][uid]['op_type']}")
            per_uid[uid] = fn
        for uid in plan["arange_uids"]:
            per_uid[uid] = wrapper.intercept_arange
        for uid in plan["position_slice_uids"]:
            per_uid[uid] = wrapper.intercept_position_slice
        for uid in plan["position_cumsum_uids"]:
            per_uid[uid] = wrapper.intercept_position_cumsum
        executor.register_op_uid_interceptors(per_uid)
        executor._decoder_kv_wrapper = wrapper
        how = ("positions from the caller" if plan["positions_input"] else
               f"{len(plan['arange_uids'])} arange, {len(plan['position_slice_uids'])} table slice, "
               f"{len(plan['position_cumsum_uids'])} count offset")
        print(f"   [{dec_name}] KV cache: {plan['num_layers']} self-attention layers cached, "
              f"{len(plan['cross_attn_uids'])} cross-attentions native, "
              f"heads={plan['num_heads']} d={plan['head_dim']} max={kv.max_cache_len}, {how}")
    wrapper.reset_for_new_sequence()
    return wrapper


class DecodeSteps:
    """The step protocol of a cached decode loop, for one sequence of the cache (a branch).

    The first run sees the whole context (the prefill, through the cache's prefill mode); every later
    run sees the one new token — `feed(whole, last)` picks — placed at its absolute position by the
    cache (`update_position_offset` before the run) and, when the decoder takes its positions from the
    caller, by the positions the flow hands with it. Without a cache (`wrapper is None`: the recompute
    oracle) every run sees the whole context, as before.
    """

    def __init__(self, wrapper: Optional[Any], branch: Optional[Any] = None):
        self.wrapper = wrapper
        self.branch = branch
        self.runs = 0

    @property
    def cached(self) -> bool:
        return self.wrapper is not None

    @property
    def one_token(self) -> bool:
        """Whether the next run sees one token (a decode step) rather than the whole context."""
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
            self.wrapper.set_decode_mode(actual_seq_len=1)
