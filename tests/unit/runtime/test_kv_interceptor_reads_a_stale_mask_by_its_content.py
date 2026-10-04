"""The KV-cache interceptor drops a mask that no longer covers the cached keys only when
the mask's own content is the causal pattern — prefill: causal with equal query and key
lengths; decode: every cached key. Any other mask is refused by name (layer, phase,
shape, lengths).

It used to drop ANY such mask and switch to causal attention by its shape alone — the
same guess removed from compiled_ops.standard_attention (a padding mask frozen at the
trace length became causal attention silently). What these tests would do if the code
were wrong: the refusal tests would not raise (seen on injection of the old drop).
"""
import pytest
import torch
import torch.nn.functional as F

from neurobrix.core.runtime.graph.kv_cache_wrapper import KVCacheAttentionWrapper, KVCacheConfig

H, D, T = 2, 4, 5            # heads, head dim, the frozen trace extent of the mask


def _wrapper():
    cfg = KVCacheConfig(num_layers=1, num_kv_heads=H, k_head_dim=D, v_head_dim=D,
                        max_cache_len=64, dtype="float32")
    return KVCacheAttentionWrapper(cfg, num_heads=H)


def _qkv(s, seed):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(1, H, s, D, generator=g) for _ in range(3))


def _mask(kind, n=T):
    tri = torch.ones(n, n, dtype=torch.bool).tril()
    if kind == "causal_bool":
        return tri
    if kind == "causal_additive":
        return torch.where(tri, 0.0, torch.finfo(torch.float32).min)
    if kind == "padding":                       # causal AND a padded first key
        m = torch.where(tri, 0.0, float("-inf")); m[:, 0] = float("-inf"); return m
    if kind == "row":                           # a [1, n] key-padding row
        return torch.zeros(1, n)
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["causal_bool", "causal_additive"])
def test_prefill_with_a_frozen_causal_mask_is_causal(kind):
    w = _wrapper(); q, k, v = _qkv(9, 0)                       # 9 tokens > frozen 5
    out = w.intercept_attention(q, k, v, attn_mask=_mask(kind), is_causal=False, layer_idx=0,
                                k_pre_transposed=False, v_pre_transposed=False)
    assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-6)


@pytest.mark.parametrize("kind", ["padding", "row"])
def test_prefill_with_a_frozen_mask_that_is_not_causal_is_refused(kind):
    w = _wrapper(); q, k, v = _qkv(9, 1)
    with pytest.raises(RuntimeError, match=r"KV interceptor \(layer 0, prefill\).*(not the causal|not a square)"):
        w.intercept_attention(q, k, v, attn_mask=_mask(kind), is_causal=False, layer_idx=0,
                              k_pre_transposed=False, v_pre_transposed=False)


def test_decode_with_a_causal_mask_attends_to_every_cached_key():
    w = _wrapper(); q, k, v = _qkv(6, 2)
    w.intercept_attention(q, k, v, attn_mask=_mask("causal_bool", 6), is_causal=False, layer_idx=0,
                          k_pre_transposed=False, v_pre_transposed=False)      # prefill 6, mask covers
    w.set_decode_mode()
    q1, k1, v1 = _qkv(1, 3)
    out = w.intercept_attention(q1, k1, v1, attn_mask=_mask("causal_bool", 1), is_causal=False,
                                layer_idx=0, k_pre_transposed=False, v_pre_transposed=False)
    kf, vf = torch.cat([k, k1], 2), torch.cat([v, v1], 2)
    assert torch.allclose(out, F.scaled_dot_product_attention(q1, kf, vf), atol=1e-6)


def test_decode_with_a_mask_that_is_not_causal_is_refused():
    """A decode step's own mask carrying a bias (not the 0 / -inf causal pattern)."""
    w = _wrapper(); q, k, v = _qkv(6, 4)
    w.intercept_attention(q, k, v, attn_mask=None, is_causal=True, layer_idx=0,
                          k_pre_transposed=False, v_pre_transposed=False)
    w.set_decode_mode()
    q1, k1, v1 = _qkv(1, 5)
    with pytest.raises(RuntimeError, match=r"KV interceptor \(layer 0, decode\)"):
        w.intercept_attention(q1, k1, v1, attn_mask=torch.full((1, 1), 0.5), is_causal=False, layer_idx=0,
                              k_pre_transposed=False, v_pre_transposed=False)


def test_a_mask_that_covers_the_keys_is_kept_untouched():
    w = _wrapper(); q, k, v = _qkv(T, 6)
    m = _mask("padding")
    out = w.intercept_attention(q, k, v, attn_mask=m, is_causal=False, layer_idx=0,
                                k_pre_transposed=False, v_pre_transposed=False)
    assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v, attn_mask=m), atol=1e-6, equal_nan=True)
