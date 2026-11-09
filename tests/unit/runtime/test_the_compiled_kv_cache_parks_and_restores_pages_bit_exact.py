"""The compiled engine's paged KV cache speaks the host tier's protocol (core/runtime/graph/paged_kv_cache.py tier_*):
its pages offloaded to host pages and restored into other pages come back bit-exact, every layer's K and V, and the
cache then holds the restored length. On the CPU (card-less) and on the machine's accelerator when there is one."""
from types import SimpleNamespace

import pytest
import torch

from nbxtensor.memory.kv_host_pages import HostPages
from nbxtensor.memory.kv_host_tier import sequence_pages
from neurobrix.core.runtime.graph.paged_kv_cache import PagedKVCache

LAYERS, H, D, TPP = 3, 2, 4, 8


def _devices():
    out = ["cpu"]
    if torch.backends.mps.is_available():
        out.append("mps")
    elif torch.cuda.is_available():
        out.append("cuda")
    return out


@pytest.mark.parametrize("device", _devices())
def test_pages_round_trip_bit_exact_through_the_host(device):
    cfg = SimpleNamespace(num_layers=LAYERS, num_kv_heads=H, k_head_dim=D, v_head_dim=D, dtype="float16",
                          max_cache_len=64, initial_cache_len=32, ceiling_is_window=False)
    cache = PagedKVCache(cfg, page_tokens=TPP, async_release=False)
    g = torch.Generator().manual_seed(3)
    with torch.inference_mode():                     # the compiled flow prefills in inference mode: its pools are
        for layer in range(LAYERS):                  # inference tensors, and the tier runs outside it (RUN 92: a 500)
            cache.update(layer, torch.randn(1, H, 24, D, generator=g).half().to(device),
                         torch.randn(1, H, 24, D, generator=g).half().to(device))
    want = [tuple(t.clone() for t in cache.get_cache(layer)) for layer in range(LAYERS)]
    layers, kb, vb = cache.tier_layout()
    host = HostPages.fixed(3, layers, kb, vb, pinned=False)
    cache.tier_offload(sequence_pages(cache.tables, cache.batch), host, [2, 0, 1])
    cache.clear()                                    # another conversation takes the cache
    with torch.inference_mode():
        for layer in range(LAYERS):
            cache.update(layer, torch.zeros(1, H, 40, D).half().to(device), torch.zeros(1, H, 40, D).half().to(device))
    cache.truncate(0)
    cache.tier_resume(24)
    cache.tier_restore(host, [2, 0, 1], sequence_pages(cache.tables, cache.batch)[:3])
    assert cache.tier_length() == 24
    for layer in range(LAYERS):
        k, v = cache.get_cache(layer)
        assert torch.equal(k, want[layer][0]) and torch.equal(v, want[layer][1])
    host.close()
