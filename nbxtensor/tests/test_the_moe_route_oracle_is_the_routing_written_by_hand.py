"""`nbxtensor.ops.oracles.fp64.moe_route_oracle`: the float64 reference of the routing op, numpy only, read against
answers written by hand. The router product is made the identity (hidden = the logits, the router weight the unit
matrix), so every score is `1 / (1 + exp(-l))` or a softmax of known logits, and the chosen experts are read off the
numbers below.

Logits of 8 experts in 4 groups of 2: (3, -3 | 1, 0.5 | 2.5, -5 | 0, -0.5). Sigmoid group scores (sum of the two):
1.000, 1.353, 0.931, 0.878, so the 2 kept groups are 1 and 0; the top 2 of their experts are 0 (0.953) and 2 (0.731).
Ungrouped, expert 4 (0.924) is second instead of 2.

What this test does if the code is wrong: the groups ignored or ranked by the wrong score, a dropped expert routed to,
the weight read from the BIASED choice value, the renormalisation or the scale skipped or swapped, the fill ignored,
n_group 1 different from the ungrouped routing - red."""
import math

import numpy as np
import pytest

from _moe_route_draw import draw
from nbxtensor.ops.oracles import fp64 as O

LOGITS = np.array([[3.0, -3.0, 1.0, 0.5, 2.5, -5.0, 0.0, -0.5]])
EYE = np.eye(8)
SIG = dict(score_fn="sigmoid", group_score="top2_sum", fill=-math.inf)


def _sig(v):
    return 1.0 / (1.0 + math.exp(-v))


def _route(bias=None, logits=LOGITS, **a):
    idx, wt, _, _ = O.moe_route_oracle(logits, np.eye(logits.shape[1]), bias, **a)()
    return dict(zip(idx[0].tolist(), wt[0].tolist()))


def test_the_kept_groups_decide_the_experts():
    got = _route(top_k=2, n_group=4, topk_group=2, renorm_eps=1e-20, scale=2.0, **SIG)
    s0, s2 = _sig(3.0), _sig(1.0)
    assert set(got) == {0, 2}
    assert got[0] == pytest.approx(2.0 * s0 / (s0 + s2), rel=1e-15)
    assert got[2] == pytest.approx(2.0 * s2 / (s0 + s2), rel=1e-15)


def test_one_group_is_the_ungrouped_routing():
    got = _route(top_k=2, n_group=1, topk_group=1, renorm_eps=None, scale=1.0, **SIG)
    assert got == pytest.approx({0: _sig(3.0), 4: _sig(2.5)}, rel=1e-15)


def test_the_group_score_max_ranks_by_the_largest():
    # group maxima 0.953, 0.731, 0.924, 0.5: the kept 2 are 0 and 2, so experts 0 and 4.
    got = _route(top_k=2, n_group=4, topk_group=2, renorm_eps=None, scale=1.0,
                 score_fn="sigmoid", group_score="max", fill=-math.inf)
    assert set(got) == {0, 4}


def test_the_bias_steers_the_choice_and_never_the_weight():
    bias = np.zeros(8)
    bias[3] = 0.5                                     # expert 3: choice 0.622 + 0.5, above expert 0's 0.953
    got = _route(bias, top_k=1, n_group=4, topk_group=1, renorm_eps=None, scale=1.0, **SIG)
    assert got == pytest.approx({3: _sig(0.5)}, rel=1e-15)


def test_a_fill_of_zero_outranks_a_negative_choice_and_minus_infinity_never_does():
    bias = np.full(8, -0.9)
    a = dict(n_group=4, topk_group=1, renorm_eps=None, scale=1.0, score_fn="sigmoid", group_score="top2_sum")
    # group scores with the bias: -0.800, -0.447, -0.869, -0.922: group 1 kept, its choices -0.169 (2), -0.278 (3).
    assert set(_route(bias, fill=-math.inf, top_k=2, **a)) == {2, 3}
    with_zero = _route(bias, fill=0.0, top_k=3, **a)  # every dropped expert at 0 outranks both: the lower ids first
    assert set(with_zero) == {0, 1, 4}
    assert with_zero[0] == pytest.approx(_sig(3.0), rel=1e-15)   # the weight is the un-biased score


def test_the_softmax_with_the_group_maximum():
    p = np.exp(LOGITS[0] - LOGITS[0].max())
    p = p / p.sum()
    got = _route(top_k=2, n_group=4, topk_group=2, renorm_eps=None, scale=16.0,
                 score_fn="softmax", group_score="max", fill=0.0)
    assert got == pytest.approx({0: 16.0 * p[0], 4: 16.0 * p[4]}, rel=1e-14)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_one_group_equals_every_group_kept_and_a_plain_top_k(seed):
    x, w, b = draw(16, 256, 64, seed, 0.0)
    a = dict(top_k=8, renorm_eps=1e-20, scale=2.5, score_fn="sigmoid", group_score="top2_sum", fill=-math.inf)
    i1, w1, _, _ = O.moe_route_oracle(x, w, b, n_group=1, topk_group=1, **a)()
    i8, w8, _, _ = O.moe_route_oracle(x, w, b, n_group=8, topk_group=8, **a)()
    assert np.array_equal(i1, i8) and np.array_equal(w1, w8)
    s = 1.0 / (1.0 + np.exp(-(x.astype(np.float64) @ w.astype(np.float64).T)))
    top = np.argsort(-(s + b), axis=-1, kind="stable")[:, :8]
    assert np.array_equal(np.sort(i1, -1), np.sort(top, -1))
    plain = np.take_along_axis(s, top, -1)
    plain = 2.5 * plain / (plain.sum(-1, keepdims=True) + 1e-20)
    assert np.allclose(np.take_along_axis(w1, np.argsort(i1, -1), -1),
                       np.take_along_axis(plain, np.argsort(top, -1), -1), rtol=1e-15, atol=0)
