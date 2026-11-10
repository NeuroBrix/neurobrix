"""The compiled engine's prefill chunk over a session's cached keys (`KVCacheAttentionWrapper._prefill_over_pages`),
with the served plan's large chunks: at 4 096 rows over 61 440 keys its dense rows x keys mask was 480 MiB in fp16 and
ended the 79 931-token turn on a full V100 with an HTTP 500 (2026-11-10), beside K and V repeated for every query head.
It attends one page of query rows at a time, each query group of a KV head on its own, over the KV heads as stored.
What this would do if the code were wrong: a served long prompt out of memory at the card's reach, or — the cut made
wrongly — a row reading keys after its own position."""
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
    assert seen
    for q_shape, k_shape, m_shape in seen:
        assert q_shape[1] == HK and k_shape[1] == HK, "K / V stay the KV heads: no expansion per query head"
        assert q_shape[2] <= PT and m_shape[0] <= PT, "one page of query rows at a time"
        assert m_shape[1] == k_shape[2] <= total, "a block reads the keys up to its last row, no further"


def test_the_mask_a_chunk_holds_is_one_page_of_rows_whatever_the_chunk():
    w = _wrapper()
    g = torch.Generator().manual_seed(1)
    rows = 8 * PT
    q, k, v = (torch.randn(1, h, rows, D, generator=g) for h in (H, HK, HK))
    w.begin_chunk(0, rows, whole_pages=True)
    w._prefill_over_pages(q, k, v, None, 0.0, D ** -0.5, 0)
    (zeros, triangle), = w._chunk_masks.values()
    assert tuple(zeros.shape) == (PT, rows) and tuple(triangle.shape) == (PT, PT)
    assert not zeros.any(), "the buffer is given back as zeros after every block"
