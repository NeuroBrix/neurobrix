"""Prompt-aware KV sizing — S1 finding TINYLLAMA-KVCAP (fixed 2026-09-01).

The run-mode KV budget (`max_tokens + prompt_margin`) was blind to the
actual prompt: any prompt beyond ~`prompt_margin` tokens overflowed at
prefill on BOTH engines while the model's window had plenty of room
(TinyLlama: 1,740-token prompt vs a 640-slot cache under a 2,048
window). Pins:

  1. Triton: the lazy layer allocation (which happens AT the prefill,
     where the prompt length is known) sizes to prompt + decode budget
     when the plan size cannot hold it, bounded by the model window;
     short prompts allocate exactly the plan size (byte-unchanged);
     no reallocation ever (the recorded-replay address contract).
  2. A prompt at/beyond the window raises the loud model-limit error.
  3. Compiled: the factory raises the growth CEILING to the window
     while the initial allocation stays at the plan size — the
     wrapper's existing on-demand growth does the rest.
"""
from __future__ import annotations

import pytest

from neurobrix.triton.dtype import NBXDtype
from neurobrix.triton.kv_cache import TritonKVCache


def _cache(plan_len=640, window=2048, budget=512):
    return TritonKVCache(
        num_layers=2, num_kv_heads=2, k_head_dim=8, v_head_dim=8,
        max_cache_len=plan_len, dtype=NBXDtype.float16,
        window_ceiling=window, decode_budget=budget)


def test_short_prompt_allocates_exactly_the_plan_size() -> None:
    c = _cache()
    assert c._alloc_len_for(13) == 640          # 13+512 <= 640
    assert c._alloc_len_for(128) == 640         # boundary: 128+512 == 640


def test_long_prompt_sizes_to_need_bounded_by_window() -> None:
    c = _cache()
    assert c._alloc_len_for(600) == 1112        # 600+512, inside window
    assert c._alloc_len_for(1740) == 2048       # min(1740+512, window)


def test_window_is_the_models_own_limit() -> None:
    c = _cache()
    with pytest.raises(RuntimeError, match="context window"):
        c._alloc_len_for(2048)
    with pytest.raises(RuntimeError, match="context window"):
        c._alloc_len_for(5000)


def test_no_window_grows_to_need() -> None:
    c = _cache(window=0)
    assert c._alloc_len_for(1740) == 2252       # 1740+512, unbounded
    assert c._alloc_len_for(13) == 640


def test_no_budget_falls_back_to_minimal_fit() -> None:
    c = _cache(budget=0)
    assert c._alloc_len_for(13) == 640
    assert c._alloc_len_for(700) == 701         # first_len+1, minimal


def test_compiled_factory_raises_ceiling_keeps_initial() -> None:
    """Path-1 composition: window > plan -> max_len=window,
    initial=plan (allocation preserved); no window -> untouched."""
    from neurobrix.core.module.cache.factory import StateCacheFactory

    class _Plan:
        pass

    class _KVPlan:
        num_layers, num_kv_heads = 2, 2
        k_head_dim, v_head_dim = 8, 8
        max_cache_len = 640
        dtype = "float16"
        initial_cache_len = 0

    class _Ctx:
        plan = _Plan()

    _Ctx.plan.kv_cache_plan = _KVPlan()
    lm = {"num_layers": 2, "num_heads": 2, "hidden_size": 16,
          "max_position_embeddings": 2048}
    w = StateCacheFactory.create(_Ctx(), lm, "cuda", "float16")
    assert w.cache.config.max_cache_len == 2048
    assert w.cache.config.initial_cache_len == 640

    _Ctx.plan.kv_cache_plan = _KVPlan()
    lm_nowin = {"num_layers": 2, "num_heads": 2, "hidden_size": 16}
    w2 = StateCacheFactory.create(_Ctx(), lm_nowin, "cuda", "float16")
    assert w2.cache.config.max_cache_len == 640
    assert w2.cache.config.initial_cache_len == 0
