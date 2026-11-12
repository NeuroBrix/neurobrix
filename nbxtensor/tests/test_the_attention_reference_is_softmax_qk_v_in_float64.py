"""`ops.oracles.attention_fp64`: the attention family's float64 reference, on query-row windows.

What this test does if the code is wrong: a window's rows that are not softmax(q k^T * scale + mask) v of THOSE rows —
red (a wrong scale, a mask not added, the softmax over the wrong axis, a window shifted); the operands not read through
the caller's `widen` — red. Card-less."""
import numpy as np

from nbxtensor.ops.oracles.attention_fp64 import attention_fp64


def _naive(q, k, v, scale, mask):
    s = np.einsum("bhqd,bhkd->bhqk", q.astype(np.float64), k.astype(np.float64)) * scale
    if mask is not None:
        s = s + mask
    p = np.exp(s - s.max(-1, keepdims=True))
    return np.einsum("bhqk,bhkd->bhqd", p / p.sum(-1, keepdims=True), v.astype(np.float64))


def test_each_window_is_the_reference_of_its_rows():
    rng = np.random.default_rng(1)
    q, k, v = (rng.standard_normal(s).astype(np.float32) for s in ((1, 2, 12, 8), (1, 2, 20, 8), (1, 2, 20, 6)))
    mask = np.where(rng.random((1, 2, 12, 20)) < 0.3, -1e9, 0.0)
    want = _naive(q, k, v, 0.35, mask)
    got = attention_fp64(q, k, v, 0.35, [(0, 3), (5, 9), (11, 12)], mask)
    assert [w for w, _ in got] == [(0, 3), (5, 9), (11, 12)]
    for (r0, r1), rows in got:
        assert np.allclose(rows, want[..., r0:r1, :], rtol=0, atol=1e-12)


def test_the_operands_are_read_through_widen():
    q = np.ones((1, 1, 2, 4), np.float32)
    seen = []
    attention_fp64(q, q, q, 1.0, [(0, 2)], widen=lambda a: seen.append(a.shape) or np.asarray(a))
    assert len(seen) == 3
