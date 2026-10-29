"""The Triton engine's paged KV cache (triton/paged_kv_cache.py) through a whole request: a prompt's prefill, then decode
steps whose append takes no host input (the device length and the block table), each step's attention against the
float64 attention over every position the sequence holds.

What each test would do if the code were wrong:
  1. an append at the wrong position or page, a length not advanced, or a read of another sequence's pages gives an
     attention that differs from the oracle's over the sequence's own K / V — red at the first wrong step;
  2. a sequence decoding past the pages reserved at prefill must be granted another page and the block table rewritten:
     a cache that does not grow writes into a page it does not hold — red (the oracle differs, or the refusal fires);
  3. the cache holds only the pages granted: its bytes are pages x page bytes, far under max_len's;
  4. clear() returns every page and zeroes the lengths: a second request starting at position 0 equals its oracle.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from nbxtensor.tensor import NBXTensor, NBXDtype, DeviceAllocator
from neurobrix.triton.paged_kv_cache import PagedTritonKVCache

BOUND = 2e-03


def _has_gpu() -> bool:
    try:
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_gpu(), reason="needs a device")


def _d2h(t):
    n = t.numel()
    sz = 2 if t._dtype == NBXDtype.float16 else 4
    buf = (ctypes.c_char * (n * sz))()
    DeviceAllocator.memcpy(ctypes.addressof(buf), t.data_ptr(), n * sz, kind=2)
    return np.frombuffer(bytes(buf), dtype=np.float16 if sz == 2 else np.float32).copy()


def _t(x):
    return NBXTensor.from_numpy(np.ascontiguousarray(x))


def _oracle(q, k, v, scale):
    """q [B, H, D], k / v [B, H_kv, T, D] -> [B, H, D] float64."""
    g = q.shape[1] // k.shape[1]
    K = np.repeat(k.astype(np.float64), g, axis=1)
    V = np.repeat(v.astype(np.float64), g, axis=1)
    s = np.einsum("bhd,bhtd->bht", q.astype(np.float64), K) * scale
    e = np.exp(s - s.max(-1, keepdims=True))
    return np.einsum("bht,bhtd->bhd", e / e.sum(-1, keepdims=True), V)


def _request(cache, rng, b, h, h_kv, d, prompt, steps, layers):
    scale = 1.0 / np.sqrt(d)
    ks = [rng.standard_normal((b, h_kv, prompt, d)).astype(np.float16) for _ in range(layers)]
    vs = [rng.standard_normal((b, h_kv, prompt, d)).astype(np.float16) for _ in range(layers)]
    for li in range(layers):
        cache.prefill(li, _t(ks[li]), _t(vs[li]))
    worst = 0.0
    for _ in range(steps):
        for li in range(layers):
            kn = rng.standard_normal((b, h_kv, 1, d)).astype(np.float16)
            vn = rng.standard_normal((b, h_kv, 1, d)).astype(np.float16)
            q = rng.standard_normal((b, h, 1, d)).astype(np.float16)
            ks[li] = np.concatenate([ks[li], kn], axis=2)
            vs[li] = np.concatenate([vs[li], vn], axis=2)
            out = cache.decode(li, _t(q), _t(kn), _t(vn), scale, 32)
            got = _d2h(out).astype(np.float64).reshape(b, h, d)
            ref = _oracle(q[:, :, 0], ks[li], vs[li], scale)
            assert np.isfinite(got).all()
            worst = max(worst, float(np.abs(got - ref).max() / np.abs(ref).max()))
    return worst


@pytest.mark.parametrize("b,h,h_kv,d,prompt,steps,budget", [
    (1, 32, 8, 128, 100, 6, 10),    # GQA 4, inside the reservation
    (2, 16, 2, 64, 63, 5, 0),       # two sequences; no decode budget: page 2 is granted mid-decode (63 + 1 = 64 fits,
                                    # 65 crosses into a new 64-position page)
])
def test_a_request_decodes_as_the_oracle_over_its_own_positions(b, h, h_kv, d, prompt, steps, budget):
    cache = PagedTritonKVCache(num_layers=2, num_kv_heads=h_kv, k_head_dim=d, v_head_dim=d, max_cache_len=4096,
                               dtype=NBXDtype.float16, page_tokens=64, async_release=False, decode_budget=budget)
    worst = _request(cache, np.random.default_rng(0), b, h, h_kv, d, prompt, steps, 2)
    assert worst < BOUND, f"relative error {worst:.2e} against float64"
    assert cache.get_cache_len() == prompt + steps


def test_the_cache_holds_its_granted_pages_only_and_clear_returns_them():
    cache = PagedTritonKVCache(num_layers=2, num_kv_heads=8, k_head_dim=128, v_head_dim=128, max_cache_len=8192,
                               dtype=NBXDtype.float16, page_tokens=64, async_release=False, decode_budget=16)
    rng = np.random.default_rng(1)
    _request(cache, rng, 1, 32, 8, 128, 200, 3, 2)
    per_layer_page = 8 * 64 * (128 + 128) * 2
    pages = -(-(200 + 16) // 64)                       # the prompt and its decode budget, rounded to pages
    assert cache.held_bytes() == pages * per_layer_page * 2
    assert cache.held_bytes() < 8192 * 8 * 256 * 2 * 2 // 10
    cache.clear()
    assert cache.held_bytes() == 0 and cache.get_cache_len() == 0
    worst = _request(cache, rng, 1, 32, 8, 128, 50, 4, 2)
    assert worst < BOUND, f"after clear(): relative error {worst:.2e}"


def test_a_request_of_another_batch_repages_the_cleared_cache():
    # `neurobrix serve`: one process, requests of any batch (supervisor 2026-10-21 12:15) — the cache re-pages, the
    # replay plans of the previous pools retire with their generation
    if not _has_gpu():
        pytest.skip("no device")
    cache = PagedTritonKVCache(num_layers=2, num_kv_heads=2, k_head_dim=64, v_head_dim=64, max_cache_len=4096,
                               dtype=NBXDtype.float16, page_tokens=64, async_release=False, decode_budget=0)
    rng = np.random.default_rng(3)
    assert _request(cache, rng, 1, 16, 2, 64, 70, 3, 2) < BOUND
    gen = cache.generation
    cache.clear()
    worst = _request(cache, rng, 2, 16, 2, 64, 63, 5, 2)             # batch 2 after batch 1, crossing a page
    assert worst < BOUND, f"after a batch change: relative error {worst:.2e}"
    assert cache.generation > gen and cache.get_cache_len() == 63 + 5
    cache.clear()
    assert _request(cache, rng, 1, 16, 2, 64, 30, 2, 2) < BOUND         # and back


def _two_devices() -> bool:
    try:
        return DeviceAllocator.device_count() >= 2
    except Exception:
        return False


@pytest.mark.skipif(not _two_devices(), reason="needs two devices")
@pytest.mark.parametrize("prompt,steps,budget", [(100, 4, 8), (63, 5, 0)])     # the second crosses a page mid-decode
def test_a_decoder_spread_over_two_cards_decodes_each_layer_on_its_own(prompt, steps, budget):
    """A plan that spreads the decoder over cards (pipeline_parallel) runs each layer's attention on that layer's
    card: its pages, length and block table must live there. One device for the whole cache made layer 1, on the
    second card, read pages on the first — Ming-Lite-Omni-1.5 triton on four V100-16GB, an illegal memory access at
    its first decode (2026-10-27). Red with one device: the fault, or the refusal of a K arriving elsewhere."""
    b, h, h_kv, d, layers = 1, 16, 2, 64, 2
    cache = PagedTritonKVCache(num_layers=layers, num_kv_heads=h_kv, k_head_dim=d, v_head_dim=d, max_cache_len=4096,
                               dtype=NBXDtype.float16, page_tokens=64, async_release=False, decode_budget=budget)
    rng = np.random.default_rng(5)
    scale = 1.0 / np.sqrt(d)

    def on(dev, x):
        DeviceAllocator.set_device(dev)
        return _t(x)
    ks = [rng.standard_normal((b, h_kv, prompt, d)).astype(np.float16) for _ in range(layers)]
    vs = [rng.standard_normal((b, h_kv, prompt, d)).astype(np.float16) for _ in range(layers)]
    for li in range(layers):
        cache.prefill(li, on(li, ks[li]), on(li, vs[li]))
    worst = 0.0
    for _ in range(steps):
        for li in range(layers):
            kn = rng.standard_normal((b, h_kv, 1, d)).astype(np.float16)
            vn = rng.standard_normal((b, h_kv, 1, d)).astype(np.float16)
            q = rng.standard_normal((b, h, 1, d)).astype(np.float16)
            ks[li] = np.concatenate([ks[li], kn], axis=2)
            vs[li] = np.concatenate([vs[li], vn], axis=2)
            out = cache.decode(li, on(li, q), on(li, kn), on(li, vn), scale, 32)
            assert out._device_idx == li
            got = _d2h(out).astype(np.float64).reshape(b, h, d)
            worst = max(worst, float(np.abs(got - _oracle(q[:, :, 0], ks[li], vs[li], scale)).max()
                                     / np.abs(_oracle(q[:, :, 0], ks[li], vs[li], scale)).max()))
    assert worst < BOUND, f"relative error {worst:.2e} against float64"
    assert cache.layer_device == [0, 1]
