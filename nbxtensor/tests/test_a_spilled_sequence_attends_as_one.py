"""A sequence whose oldest pages were spilled to host memory (point 10, the mid-decode spill): the host's partial over
its pages, in the split kernel's form (unnormalised acc, running max m, sum l, natural exp), merged by log-sum-exp with
the card's partial, equals attention over the whole sequence in one piece (fp64 oracle). What this test would do if the
code were wrong: a normalised acc, an exp2 / log2e mismatch, a GQA head mapped to the wrong KV head, pages read in the
wrong order or past the spilled length, an empty host part not neutral, a layout not float32 (red)."""
import numpy as np
import pytest

from nbxtensor.ops.host_decode_partial import host_decode_partial, merge_partials


def _host(kv, cap):
    # [T, H_kv, D] -> [H_kv, cap, D] float32, the spill's layout, zero past T
    out = np.zeros((kv.shape[1], cap, kv.shape[2]), np.float32)
    out[:, :len(kv)] = kv.transpose(1, 0, 2)
    return out


def _pages(kv, page_tokens):
    # [T, H_kv, D] -> [pages, H_kv, page_tokens, D], the card pool's layout, zero past T
    t, hk, d = kv.shape
    n = -(-t // page_tokens)
    out = np.zeros((n, hk, page_tokens, d), dtype=kv.dtype)
    for p in range(n):
        chunk = kv[p * page_tokens:(p + 1) * page_tokens]
        out[p, :, :len(chunk)] = chunk.transpose(1, 0, 2)
    return out


def _full(q, k, v, groups, scale):
    # fp64 oracle: q [H, D], k / v [T, H_kv, D] -> [H, D_v]
    q, k, v = (x.astype(np.float64) for x in (q, k, v))
    out = np.empty((q.shape[0], v.shape[2]))
    for h in range(q.shape[0]):
        s = k[:, h // groups] @ q[h] * scale
        p = np.exp(s - s.max())
        out[h] = p @ v[:, h // groups] / p.sum()
    return out


def _card(q, k, v, groups, scale):
    # the card's partial in the split kernel's form, computed here in numpy: acc, m, l per query row
    s = np.einsum("hd,thd->ht", q, np.repeat(k, groups, axis=1)) * scale
    m = s.max(axis=1)
    p = np.exp(s - m[:, None])
    return np.einsum("ht,thd->hd", p, np.repeat(v, groups, axis=1)), m, p.sum(axis=1)


@pytest.mark.parametrize("total, spilled, page_tokens, heads, kv_heads", [
    (100, 32, 16, 8, 2), (100, 48, 16, 8, 8), (257, 128, 64, 32, 4), (64, 64, 16, 4, 1), (40, 0, 16, 4, 2)])
def test_the_host_partial_merged_with_the_card_s_equals_one_piece(total, spilled, page_tokens, heads, kv_heads):
    rng = np.random.default_rng(total + spilled)
    d, groups, scale = 64, heads // kv_heads, 64 ** -0.5
    q = rng.standard_normal((heads, d)).astype(np.float32)
    k = rng.standard_normal((total, kv_heads, d)).astype(np.float32)
    v = rng.standard_normal((total, kv_heads, d)).astype(np.float32)
    k_host, v_host = _host(k[:spilled], spilled + page_tokens), _host(v[:spilled], spilled + page_tokens)
    host = host_decode_partial(q, k_host, v_host, spilled, groups, scale)
    card = _card(q, k[spilled:], v[spilled:], groups, scale) if spilled < total else None
    out = merge_partials([card, host] if card is not None else [host])
    assert np.allclose(out, _full(q, k, v, groups, scale), rtol=0, atol=2e-5)


def test_an_empty_host_part_is_neutral_and_refusals_are_named():
    q = np.ones((4, 8), np.float32)
    k = np.zeros((2, 16, 8), np.float32)
    o, m, l = host_decode_partial(q, k, k, 0, 2, 1.0)
    assert np.all(m == -np.inf) and np.all(l == 0) and np.all(o == 0)
    with pytest.raises(ValueError, match="pages"):
        host_decode_partial(q, k, k, 17, 2, 1.0)              # more tokens than the host pages hold
    with pytest.raises(ValueError, match="heads"):
        host_decode_partial(q, k, k, 4, 3, 1.0)               # 4 query heads are not 2 KV heads x 3
    with pytest.raises(ValueError, match="float32"):
        host_decode_partial(q, k.astype(np.float16), k.astype(np.float16), 4, 2, 1.0)   # a layout converted per step


@pytest.mark.parametrize("total, spilled, page_tokens, heads, kv_heads", [(300, 128, 64, 8, 2), (130, 64, 64, 4, 4)])
def test_the_card_s_paged_decode_takes_the_host_partial_as_its_last_split(total, spilled, page_tokens, heads, kv_heads):
    """On the accelerator: the card's pages (the newest positions) through the paged split kernel, the spilled ones'
    partial from the host as the reduce's last split — equal to one piece."""
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.memory.allocator import DeviceAllocator
    from nbxtensor.ops.attention_wrappers import paged_decode_attention
    from nbxtensor.tensor import NBXTensor
    if DeviceAllocator.device_count() == 0:          # only a machine without an accelerator skips; any other error fails
        pytest.skip("no accelerator on this machine")
    rng = np.random.default_rng(total)
    d, groups, scale = 64, heads // kv_heads, 64 ** -0.5
    q = rng.standard_normal((heads, d)).astype(np.float32)
    k = rng.standard_normal((total, kv_heads, d)).astype(np.float32)
    v = rng.standard_normal((total, kv_heads, d)).astype(np.float32)
    host = host_decode_partial(q, _host(k[:spilled], spilled), _host(v[:spilled], spilled), spilled, groups, scale)
    card_k, card_v = _pages(k[spilled:], page_tokens), _pages(v[spilled:], page_tokens)
    order = rng.permutation(len(card_k))                           # the card's pages anywhere in its pool
    pool_k, pool_v = np.zeros_like(card_k), np.zeros_like(card_v)
    pool_k[order], pool_v[order] = card_k, card_v
    dev = lambda a, dt=None: NBXTensor.from_numpy(a).to(device="cuda:0")          # noqa: E731
    out = paged_decode_attention(dev(q.reshape(1, heads, 1, d)), dev(pool_k), dev(pool_v),
                                 dev(order.astype(np.int32).reshape(1, -1)), dev(np.array([total - spilled], np.int32)),
                                 scale, total - spilled, host_partial=host)
    assert np.allclose(out.numpy().reshape(heads, d), _full(q, k, v, groups, scale), rtol=0, atol=2e-5)
