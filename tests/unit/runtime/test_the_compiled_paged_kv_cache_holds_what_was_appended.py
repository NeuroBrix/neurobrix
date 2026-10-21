"""The compiled engine's paged KV cache (core/runtime/graph/paged_kv_cache.py) — what its attention reads.

The cache returns, at every update, the layer's whole K / V gathered through the page tables. What each test would do
if the code were wrong:
  1. a position written to the wrong page or slot, or a gather in the wrong page order, returns K / V that differ from
     the plain concatenation of everything appended — the comparison is exact (a copy changes no bit);
  2. a prompt longer than the pool's starting size must grow the pool (PagePool.grow, the layers' pools extended): a
     cache that does not grow raises StateExhausted, and one that grows wrongly returns other values;
  3. a request past the ceiling is refused by name, never truncated;
  4. clear() returns every page, and the next request starts at position 0 with its own values.
Runs on the CPU: the cache's device is the incoming tensor's.
"""
from __future__ import annotations

import pytest
import torch

from neurobrix.core.runtime.graph.kv_cache_wrapper import KVCacheConfig
from neurobrix.core.runtime.graph.paged_kv_cache import PagedKVCache


def _cfg(max_len=512, initial=0):
    return KVCacheConfig(num_layers=2, num_kv_heads=4, k_head_dim=16, v_head_dim=8, max_cache_len=max_len,
                         dtype="float32", initial_cache_len=initial, page_tokens=16, async_release=False)


def _request(cache, b, prompt, steps, g):
    ref_k = [None, None]
    ref_v = [None, None]
    for t in [prompt] + [1] * steps:
        for layer in range(2):
            k = torch.randn(b, 4, t, 16, generator=g)
            v = torch.randn(b, 4, t, 8, generator=g)
            ref_k[layer] = k if ref_k[layer] is None else torch.cat([ref_k[layer], k], 2)
            ref_v[layer] = v if ref_v[layer] is None else torch.cat([ref_v[layer], v], 2)
            got_k, got_v = cache.update(layer, k, v)
            assert torch.equal(got_k, ref_k[layer]) and torch.equal(got_v, ref_v[layer])
    return ref_k


def test_every_update_returns_exactly_the_positions_appended():
    cache = PagedKVCache(_cfg(), 16, False)
    _request(cache, 2, 37, 30, torch.Generator().manual_seed(0))       # crosses page boundaries during decode
    assert cache.get_seq_len() == 67
    k, v = cache.get_cache(1)
    assert k.shape == (2, 4, 67, 16) and v.shape == (2, 4, 67, 8)


def test_a_prompt_past_the_starting_size_grows_the_pool():
    cache = PagedKVCache(_cfg(max_len=512, initial=32), 16, False)
    _request(cache, 1, 100, 3, torch.Generator().manual_seed(1))       # starts at 2 pages, needs 7
    assert cache.pool.num_pages >= 7
    assert all(p.shape[0] == cache.pool.num_pages for p in cache.k_pools.values())


def test_a_request_past_the_ceiling_is_refused_by_name():
    cache = PagedKVCache(_cfg(max_len=64), 16, False)
    with pytest.raises(RuntimeError, match="exceed the plan's cache"):
        cache.update(0, torch.zeros(1, 4, 65, 16), torch.zeros(1, 4, 65, 8))


def test_clear_returns_every_page_and_the_next_request_starts_at_zero():
    cache = PagedKVCache(_cfg(), 16, False)
    g = torch.Generator().manual_seed(2)
    _request(cache, 1, 40, 5, g)
    assert cache.held_bytes() > 0
    cache.clear()
    assert cache.held_bytes() == 0 and cache.get_seq_len() == 0
    _request(cache, 1, 9, 4, g)


def test_a_request_of_another_batch_repages_the_cleared_cache():
    # `neurobrix serve`: one process, requests of any batch (supervisor 2026-10-21 12:15) — the cache re-pages
    cache = PagedKVCache(_cfg(), 16, False)
    _request(cache, 1, 40, 10, torch.Generator().manual_seed(0))
    cache.clear()
    _request(cache, 2, 37, 30, torch.Generator().manual_seed(1))      # batch 2 after batch 1: exact, page-crossing
    assert cache.get_seq_len() == 67
    cache.clear()
    _request(cache, 1, 20, 5, torch.Generator().manual_seed(2))       # and back


def test_a_batch_change_mid_request_is_refused_by_name():
    cache = PagedKVCache(_cfg(), 16, False)
    _request(cache, 1, 10, 2, torch.Generator().manual_seed(0))
    with pytest.raises(RuntimeError, match="mid-request"):
        cache.update(0, torch.randn(2, 4, 1, 16), torch.randn(2, 4, 1, 8))
