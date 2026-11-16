"""How the routing op's device tests judge a launch against the float64 oracle, in one place so a host test can prove
that the judgement itself goes red.

Per token: the indices as a SET (the slot order is not the contract), the weights paired by expert id.
- Equal sets: the weights are compared with the oracle's, relative to the token's largest weight.
- Different sets: allowed ONLY on a near-tie, that is a reference gap ABOVE 0 and at most the band (two choice bounds;
  the gap between the k-th and (k+1)-th choice values, or half the gap between the last kept and first dropped group
  score). An EXACT tie (gap 0: the lanes of a dropped group all hold the fill) is not a near-tie: the op's rule (lower
  id first) fixes the set and the reference set is required. On a near-tie the weights are still compared, on the ids
  the device chose: the reference weights of THAT set are computed from the float64 scores (renormalised over the
  device's set), so a wrong weight cannot hide behind a different set.
The near-tie reading uses the reference alone, so a wrong kernel cannot widen it."""
import math

import numpy as np


def gap_of(v, n):
    """The gap between the n-th and (n+1)-th largest values of a row (inf when there is no (n+1)-th)."""
    srt = np.sort(v)[::-1]
    return math.inf if n >= len(srt) else float(srt[n - 1] - srt[n])


def scores_of(x, w, a):
    """The float64 scores (before the bias) of one token row: sigmoid or softmax of the router product."""
    z = w @ x
    if a["score_fn"] == "sigmoid":
        return 1.0 / (1.0 + np.exp(-z))
    e = np.exp(z - z.max())
    return e / e.sum()


def weights_of_set(ids, score, a):
    """The reference weights a set of expert ids gets: its scores, renormalised when asked, times the scale."""
    s = score[np.asarray(ids)]
    if a["renorm_eps"] is not None:
        s = s / (s.sum() + a["renorm_eps"])
    return s * a["scale"]


def compare(idx, wt, ref, a, name, xr, wr, weights_bound, choice_bound):
    """The worst weight deviation (relative to the largest weight of the token) and the near-tie tokens (token, gap).
    Raises AssertionError on a wrong set, a set with a repeated or out-of-range id, or an exact tie not resolved as the
    reference does. The caller asserts the weight bound."""
    ri, rw, rc, rg = ref
    band = 2 * choice_bound
    worst, ties = 0.0, []
    for t in range(idx.shape[0]):
        assert len(set(idx[t].tolist())) == a["top_k"] and idx[t].min() >= 0 and idx[t].max() < rc.shape[1], \
            f"{name} token {t}: indices {idx[t].tolist()} are not {a['top_k']} distinct experts"
        if set(idx[t].tolist()) == set(ri[t].tolist()):
            mine, theirs = wt[t][np.argsort(idx[t])], rw[t][np.argsort(ri[t])]
        else:
            gc = gap_of(rc[t], a["top_k"])
            gg = gap_of(rg[t], a["topk_group"]) / 2 if a["n_group"] > 1 else math.inf
            gap = min(gc, gg)
            assert 0 < gap <= band, (f"{name} token {t}: set {sorted(idx[t].tolist())} against "
                                     f"{sorted(ri[t].tolist())}, reference gap {gap} is "
                                     f"{'an exact tie: lower id first fixes the set' if gap == 0 else 'above the band'}")
            ties.append((t, gap))
            ids = np.sort(idx[t])
            mine, theirs = wt[t][np.argsort(idx[t])], weights_of_set(ids, scores_of(xr[t], wr, a), a)
        worst = max(worst, float(np.abs(mine - theirs).max() / np.abs(theirs).max()))
    return worst, ties
