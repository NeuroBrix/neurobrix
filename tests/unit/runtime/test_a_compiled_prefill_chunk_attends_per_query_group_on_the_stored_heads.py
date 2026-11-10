"""The compiled engine's prefill chunk over a session's cached keys (`KVCacheAttentionWrapper._prefill_over_pages`).
It repeated K and V for every query head (groups x the cached keys, about 1 GB at a V100's reach, at every layer of
every pass); it now runs one attention per query group of a KV head, on the KV heads as stored, the chunk's rows whole
— cutting the rows too (256 calls a layer) measured 141.9 s against 44.2 s on a 19 931-token turn (2026-11-10). Its
dense [rows, keys] mask is the session chunk's priced reserve and is written in place.
What this would do if the code were wrong: K / V copied per head again, a row reading keys after its own position, or
a mask that lets a chunk read the page padding past the true end."""
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from neurobrix.core.runtime.graph.kv_cache_wrapper import KVCacheAttentionWrapper, KVCacheConfig  # noqa: E402

LAYERS, HK, H, D, PT = 2, 2, 8, 32, 16


def _wrapper():
    w = KVCacheAttentionWrapper(KVCacheConfig(num_layers=LAYERS, num_kv_heads=HK, k_head_dim=D, v_head_dim=D,
                                              max_cache_len=100000, dtype="float32", initial_cache_len=PT,
                                              page_tokens=PT))
    w._num_heads, w._gqa_group_size = H, H // HK
    return w


def _oracle(q, k, v, start):
    """Causal attention of the chunk's rows over every position so far, fp64, K / V repeated per query head."""
    qq, kk, vv = (x.double()[0] for x in (q, k, v))
    kk, vv = kk.repeat_interleave(H // HK, 0), vv.repeat_interleave(H // HK, 0)
    rows, keys = qq.shape[1], kk.shape[1]
    causal = torch.arange(keys)[None, :] <= (start + torch.arange(rows))[:, None]
    s = (qq @ kk.transpose(1, 2)) * D ** -0.5
    return torch.softmax(s.masked_fill(~causal, float("-inf")), -1) @ vv


@pytest.mark.parametrize("chunk", [PT, 3 * PT, 3 * PT + 5])
def test_every_chunk_equals_causal_attention_over_every_position_so_far(monkeypatch, chunk):
    g = torch.Generator().manual_seed(chunk)
    total = 2 * chunk + PT
    q, k, v = (torch.randn(1, h, total, D, generator=g) for h in (H, HK, HK))
    w = _wrapper()
    seen, real = [], F.scaled_dot_product_attention

    def spy(qq, kk, vv, attn_mask=None, **kw):
        seen.append((tuple(qq.shape), tuple(kk.shape), tuple(attn_mask.shape)))
        return real(qq, kk, vv, attn_mask=attn_mask, **kw)
    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    for a in range(0, total, chunk):
        b = min(a + chunk, total)
        w.begin_chunk(a, b - a, whole_pages=True)
        for layer in range(LAYERS):
            got = w._prefill_over_pages(q[:, :, a:b], k[:, :, a:b], v[:, :, a:b], None, 0.0, D ** -0.5, layer)
            want = _oracle(q[:, :, a:b], k[:, :, :b], v[:, :, :b], a)
            assert got.shape == (1, H, b - a, D)
            assert torch.allclose(got[0].double(), want, atol=1e-5), (a, layer, (got[0].double() - want).abs().max())
    assert len(seen) == (H // HK) * LAYERS * len(range(0, total, chunk)), "one attention per query group, layer and chunk"
    for q_shape, k_shape, m_shape in seen:
        assert q_shape[1] == HK and k_shape[1] == HK, "K / V stay the KV heads: no expansion per query head"
        assert m_shape == (q_shape[2], k_shape[2]) and k_shape[2] % PT == 0, "the chunk's rows whole, over whole pages"


def test_the_chunk_s_mask_is_causal_at_the_cached_length_and_closes_the_page_padding():
    w = _wrapper()
    start, rows, keys = 2 * PT + 3, PT + 5, 4 * PT
    m = w._chunk_mask(torch.device("cpu"), torch.float32, start, rows, keys)
    allowed = torch.arange(keys)[None, :] <= (start + torch.arange(rows))[:, None]
    assert tuple(m.shape) == (rows, keys) and m.dtype == torch.float32
    assert (m[allowed] == 0).all() and torch.isneginf(m[~allowed]).all()
    assert w._chunk_mask(torch.device("cpu"), torch.float32, start, rows, keys) is m, "one per device and chunk"
