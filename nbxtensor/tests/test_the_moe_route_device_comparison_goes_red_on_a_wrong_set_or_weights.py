"""The judgement the device test applies to a launch (`_moe_route_compare.compare`) goes RED, on a host, on the tokens
the fill-0 cases were built for: a token whose choice values tie exactly at the fill, where the op's rule (lower id
first) fixes the set. A wrong set, wrong weights on the right set, and wrong weights under a near-tie set all fail; the
reference itself and a set swapped inside a real near-tie pass."""
import math

import numpy as np
import pytest

from _moe_route_compare import compare, scores_of, weights_of_set
from _moe_route_draw import draw
from nbxtensor.ops.oracles import fp64 as O

WEIGHTS, CHOICE = 4e-6, 1e-6
A = dict(score_fn="sigmoid", group_score="top2_sum", renorm_eps=1e-20, top_k=8, n_group=8, topk_group=2,
         fill=0.0, scale=1.0)


@pytest.fixture(scope="module")
def routed():
    """Every kept choice value below 0: every token picks 8 fill lanes (an exact tie on 0), lower id first."""
    x, w, b = draw(8, 64, 1000, 41, -1.5)
    x, w = x.astype(np.float64), w.astype(np.float64)
    ref = O.moe_route_oracle(x, w, b, **A)()
    assert (ref[2][np.arange(8)[:, None], ref[0]] == 0.0).all(), "the draw must route every token to filled experts"
    return x, w, ref


def _judge(routed, idx, wt):
    x, w, ref = routed
    return compare(idx, wt, ref, A, "case", x, w, WEIGHTS, CHOICE)


def test_the_reference_itself_passes_with_its_slots_in_any_order(routed):
    ri, rw = routed[2][0], routed[2][1]
    worst, ties = _judge(routed, ri[:, ::-1].copy(), rw[:, ::-1].copy())
    assert worst == 0.0 and ties == []


def test_a_wrong_set_on_an_exact_tie_is_refused(routed):
    ri, rw = routed[2][0].copy(), routed[2][1].copy()
    fills = np.flatnonzero(routed[2][2][0] == 0.0)          # another filled lane, also tied at 0: a different set
    ri[0, 0] = [e for e in fills if e not in ri[0]][0]
    with pytest.raises(AssertionError, match="exact tie: lower id first fixes the set"):
        _judge(routed, ri, rw)


def test_wrong_weights_on_the_right_set_are_seen(routed):
    ri, rw = routed[2][0], routed[2][1].copy()
    rw[3, 0] *= 1.001
    worst, _ = _judge(routed, ri, rw)
    assert worst > WEIGHTS


def test_a_near_tie_set_is_accepted_only_with_the_weights_of_its_own_set():
    x, w, b = draw(32, 64, 1000, 41, 0.0)
    x, w = x.astype(np.float64), w.astype(np.float64)
    a = dict(A, fill=-math.inf)
    ri, rw, rc, rg = O.moe_route_oracle(x, w, b, **a)()
    rc, t = rc.copy(), 0
    srt = np.argsort(-rc[t], kind="stable")
    kth, nxt = srt[a["top_k"] - 1], srt[a["top_k"]]
    rc[t, nxt] = rc[t, kth] - 0.5 * CHOICE                  # a doctored reference row: a gap of half a bound
    idx = ri.copy()
    idx[t, list(ri[t]).index(kth)] = nxt                    # the device picked the other of the pair
    good = rw.copy()
    good[t] = weights_of_set(idx[t], scores_of(x[t], w, a), a)
    worst, ties = compare(idx, good, (ri, rw, rc, rg), a, "near", x, w, WEIGHTS, CHOICE)
    assert [k for k, _ in ties] == [t] and worst <= WEIGHTS
    bad = good.copy()
    bad[t, 0] *= 1.001                                      # the right set, a wrong weight: red
    worst, _ = compare(idx, bad, (ri, rw, rc, rg), a, "near", x, w, WEIGHTS, CHOICE)
    assert worst > WEIGHTS
