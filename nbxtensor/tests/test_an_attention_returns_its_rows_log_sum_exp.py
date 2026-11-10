"""scaled_dot_product_attention_wrapper(return_lse=True): every route returns, beside its output, each query row's
natural-log log-sum-exp over its keys [B, H, T_q] fp32 (scaled and masked scores; -inf for a fully masked row) — what
a caller merging this attention with another over other keys needs (the KV spill's prefill chunk, point 10 step 3).
Each route against an fp64 oracle, forced where the dispatch would not take it here; then the merge of two key halves
by their LSE equals the attention over all keys. What this test would do if the code were wrong: a log2 LSE, the LSE of
the unscaled or unmasked scores, a GQA head's LSE taken from the wrong row, the rounded rows left in, a fully masked
row NaN (red)."""
import math

import numpy as np
import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.ops import attention_wrappers as A
from nbxtensor.tensor import NBXTensor


@pytest.fixture(autouse=True)
def _accelerator():
    if DeviceAllocator.device_count() == 0:          # only a machine without an accelerator skips; any other error fails
        pytest.skip("no accelerator on this machine")


def _dev(a):
    return NBXTensor.from_numpy(np.ascontiguousarray(a)).to(device="cuda:0")


def _oracle(q, k, v, mask, scale):
    q, k, v = (x.astype(np.float64) for x in (q, k, v))
    g = q.shape[1] // k.shape[1]
    k, v = np.repeat(k, g, axis=1), np.repeat(v, g, axis=1)
    s = np.einsum("bhqd,bhkd->bhqk", q, k) * scale
    if mask is not None:
        s = s + mask.astype(np.float64)
    m = s.max(-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.exp(s - np.where(np.isneginf(m), 0, m))
        lse = np.log(p.sum(-1)) + np.where(np.isneginf(m), 0, m)[..., 0]
        out = np.nan_to_num(np.einsum("bhqk,bhkd->bhqd", p, v) / p.sum(-1, keepdims=True))
    return out, lse


def _case(seed, B=1, H=8, HK=2, TQ=37, TK=150, D=64, masked_row=False, dtype=np.float16):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, H, TQ, D)).astype(dtype)
    k = rng.standard_normal((B, HK, TK, D)).astype(dtype)
    v = rng.standard_normal((B, HK, TK, D)).astype(dtype)
    mask = np.where(rng.random((TQ, TK)) < 0.2, -np.inf, 0.0).astype(np.float32)
    mask[:, 0] = 0.0
    if masked_row:
        mask[3] = -np.inf
    return q, k, v, mask


ROUTES = {
    "dispatch": lambda q, k, v, m, s: A.scaled_dot_product_attention_wrapper(q, k, v, attn_mask=m, scale=s, k_pre_transposed=False, return_lse=True),
    "math": lambda q, k, v, m, s: A._math_attention(q, k, v, attn_mask=m, scale=s, return_lse=True),
    "chunked": lambda q, k, v, m, s: A._math_attention_chunked(q, k, v, m, False, s, 8, return_lse=True),
}


@pytest.mark.parametrize("route", sorted(ROUTES) + ["flash"])
@pytest.mark.parametrize("shape", [dict(), dict(D=72), dict(H=4, HK=4, TQ=64, TK=64), dict(masked_row=True)])
def test_each_route_returns_the_rows_log_sum_exp(route, shape, monkeypatch):
    if route == "flash":                 # the dispatch forced onto its flash kernels (certified, else the forward one)
        from nbxtensor.ops import launch_keys
        monkeypatch.setattr(launch_keys, "sdpa_route", lambda *a, **kw: ("flash", 0))
        route = "dispatch"
    q, k, v, mask = _case(7, **shape)
    scale = 1 / math.sqrt(q.shape[-1])
    out, lse = ROUTES[route](_dev(q), _dev(k), _dev(v), _dev(mask), scale)
    want_out, want_lse = _oracle(q, k, v, mask, scale)
    got = lse.numpy()
    assert got.shape == want_lse.shape and lse.nbx_dtype == NBXDtype.float32
    assert np.array_equal(np.isneginf(got), np.isneginf(want_lse)) and not np.isnan(got).any()
    fin = np.isfinite(want_lse)
    assert np.allclose(got[fin], want_lse[fin], atol=2e-3, rtol=0), np.abs(got[fin] - want_lse[fin]).max()
    assert np.allclose(out.numpy().astype(np.float64), want_out, atol=3e-3)


def test_two_halves_merged_by_their_lse_equal_the_whole():
    q, k, v, _ = _case(11, TK=200)
    scale = 1 / math.sqrt(q.shape[-1])
    parts = [A.scaled_dot_product_attention_wrapper(_dev(q), _dev(k[:, :, a:b]), _dev(v[:, :, a:b]), scale=scale, k_pre_transposed=False,
                                                     return_lse=True) for a, b in ((0, 80), (80, 200))]
    (o1, l1), (o2, l2) = [(o.numpy().astype(np.float64), l.numpy().astype(np.float64)) for o, l in parts]
    m = np.maximum(l1, l2)
    w1, w2 = np.exp(l1 - m)[..., None], np.exp(l2 - m)[..., None]
    merged = (o1 * w1 + o2 * w2) / (w1 + w2)
    assert np.allclose(merged, _oracle(q, k, v, None, scale)[0], atol=3e-3)
