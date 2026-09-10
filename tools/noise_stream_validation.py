#!/usr/bin/env python3
"""The Triton branch's noise stream, validated against the distribution it claims.

WHY THIS EXISTS
---------------
The vendor-correctness cell gates a diffusion family by pinning the initial
latent: our engine renders, dumps its starting noise, and the vendor denoises
that exact tensor. That is the right primary gate — it turns PSNR back into a
real bound — but it has a blind spot that no campaign row will ever reach: a
pinned latent is OUR latent handed to THEM, so the stream that produced it is
never itself under test. Since the Triton branch draws its noise from our own
kernel (`kernels/ops/rand_op.py::randn_kernel`, seeded by
`kernels/rng_stream.py`), the justness of that stream is proved by nothing.

This tool proves it once, separately, and STATISTICALLY — not pixel to pixel,
because there is no reference stream to be pixel-equal to. It is an engine
fact, recorded as such: not a campaign row, not a per-release gate.

WHAT IS UNDER TEST
------------------
The stream, not Triton's Philox. Three things are ours and can each be wrong
while `tl.randn` is perfect:

  1. the per-draw kernel seed — `splitmix64(run_seed, counter)` truncated to
     31 bits in `rng_stream.next_seed()`;
  2. the offset mapping — every element of a draw takes the global element
     index as its Philox offset (`rand_op.py`);
  3. the composition of the two across a run: successive draws of one run, and
     runs whose seeds are neighbours.

BOUNDS ARE WRITTEN HERE, BEFORE THE MEASUREMENT
-----------------------------------------------
Each bound is 4 standard errors of the statistic's own sampling distribution
under the null "the sample is n i.i.d. draws from N(0,1)". Four sigma, not
three: with eight statistics per run a three-sigma bound would cry wolf on a
correct generator about once in forty runs, which is the failure mode this
repository has already paid for.

    mean            se = 1/sqrt(n)
    variance        se = sqrt(2/n)
    skewness        se = sqrt(6/n)
    excess kurtosis se = sqrt(24/n)
    correlation     se = 1/sqrt(n)          (autocorr, cross-draw, cross-seed)
    KS statistic    critical = 1.63/sqrt(n) (asymptotic, 1 % level)

The seed-collision check has no sigma: it counts exact collisions among the
per-draw seeds of a run and states them against the birthday expectation
n_draws^2 / 2^32 for a 31-bit space. A count near the expectation means the
truncation behaves like a random map — which is the finding either way, since
two draws sharing a seed AND an offset range are bit-identical noise.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

SIGMA = 4.0


def bounds(n: int) -> dict:
    """The bound for every statistic, from n alone. Written before any draw."""
    return {
        "mean": SIGMA / math.sqrt(n),
        "variance": SIGMA * math.sqrt(2.0 / n),
        "skewness": SIGMA * math.sqrt(6.0 / n),
        "excess_kurtosis": SIGMA * math.sqrt(24.0 / n),
        "correlation": SIGMA / math.sqrt(n),
        "ks_statistic": 1.63 / math.sqrt(n),
    }


def draw(n: int, run_seed: int, count: int, device: str) -> list:
    """`count` successive draws of n elements from ONE armed run stream.

    Consumption order is the contract: draw i is the (i+1)-th stochastic draw a
    run makes, exactly as a flow handler would consume it.
    """
    from neurobrix.kernels import rng_stream
    from neurobrix.kernels.wrappers import randn_wrapper

    rng_stream.set_run_seed(run_seed)
    out = []
    for _ in range(count):
        t = randn_wrapper([n], device=device)
        out.append(np.asarray(t.numpy(), dtype=np.float64).ravel())
    rng_stream.set_run_seed(None)
    return out


def seed_sequence(run_seed: int, count: int) -> list:
    """The per-draw seeds a run would use, drawn from the stream itself."""
    from neurobrix.kernels import rng_stream

    rng_stream.set_run_seed(run_seed)
    seeds = [rng_stream.next_seed() for _ in range(count)]
    rng_stream.set_run_seed(None)
    return seeds


def check(name: str, value: float, bound: float, detail: str = "") -> dict:
    return {"check": name, "value": float(value), "bound": float(bound),
            "verdict": "PASS" if abs(value) <= bound else "FAIL", "detail": detail}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1 << 22,
                    help="elements per draw (default 4194304)")
    ap.add_argument("--draws", type=int, default=4,
                    help="successive draws of one run to test for independence")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lags", type=int, default=8,
                    help="autocorrelation lags tested within a draw")
    ap.add_argument("--collision-draws", type=int, default=1_000_000,
                    help="length of the run whose per-draw seeds are checked for collisions")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    b = bounds(a.n)
    results = []

    samples = draw(a.n, a.seed, a.draws, a.device)

    # R33: the Triton branch draws its noise without torch. Stated first, as the
    # rule requires, and asserted rather than believed.
    torch_loaded = "torch" in sys.modules
    results.append({"check": "r33_no_torch_in_the_noise_path", "value": torch_loaded,
                    "bound": False, "verdict": "FAIL" if torch_loaded else "PASS",
                    "detail": "sys.modules after arming the stream and drawing"})

    x = samples[0]
    results.append(check("mean", float(x.mean()), b["mean"], "of draw 0"))
    results.append(check("variance_minus_1", float(x.var(ddof=1)) - 1.0, b["variance"], "of draw 0"))
    results.append(check("skewness", float(stats.skew(x)), b["skewness"], "of draw 0"))
    results.append(check("excess_kurtosis", float(stats.kurtosis(x, fisher=True)),
                         b["excess_kurtosis"], "of draw 0"))

    ks = stats.kstest(x, "norm")
    results.append(check("ks_statistic_vs_N01", float(ks.statistic), b["ks_statistic"],
                         f"p={ks.pvalue:.3g}, asymptotic 1 % critical value"))

    # Anderson-Darling is more sensitive in the tails than KS, where a Box-Muller
    # defect would show first. It is run on a subsample because its statistic is
    # not calibrated for n in the millions.
    sub = x[:: max(1, a.n // 5000)][:5000]
    ad = stats.anderson(sub, dist="norm")
    ad_crit = float(ad.critical_values[list(ad.significance_level).index(1.0)]) \
        if 1.0 in list(ad.significance_level) else float(ad.critical_values[-1])
    results.append({"check": "anderson_darling_tails", "value": float(ad.statistic),
                    "bound": ad_crit,
                    "verdict": "PASS" if ad.statistic <= ad_crit else "FAIL",
                    "detail": f"n={len(sub)} subsample, 1 % critical value"})

    # Within a draw: element i and element i+k take Philox offsets i and i+k.
    for k in range(1, a.lags + 1):
        r = float(np.corrcoef(x[:-k], x[k:])[0, 1])
        results.append(check(f"autocorrelation_lag_{k}", r, b["correlation"], "within draw 0"))

    # Between successive draws of ONE run: counter c against counter c+1. This is
    # where a weak splitmix64 mixing would surface.
    for i in range(len(samples) - 1):
        r = float(np.corrcoef(samples[i], samples[i + 1])[0, 1])
        results.append(check(f"cross_draw_corr_{i}_vs_{i+1}", r, b["correlation"],
                             "successive draws of one run"))
    if len(samples) > 2:
        r = float(np.corrcoef(samples[0], samples[-1])[0, 1])
        results.append(check(f"cross_draw_corr_0_vs_{len(samples)-1}", r, b["correlation"],
                             "first against last draw of the run"))

    # Neighbouring run seeds: the same draw index, seeds s and s+1. A user who
    # runs --seed 42 and --seed 43 must get two independent images.
    neighbour = draw(a.n, a.seed + 1, 1, a.device)[0]
    r = float(np.corrcoef(samples[0], neighbour)[0, 1])
    results.append(check(f"cross_seed_corr_{a.seed}_vs_{a.seed+1}", r, b["correlation"],
                         "draw 0 of two neighbouring run seeds"))

    # The 31-bit truncation in rng_stream.next_seed(). Two draws that collide on
    # the seed AND cover the same offset range are bit-identical noise.
    seeds = seed_sequence(a.seed, a.collision_draws)
    distinct = len(set(seeds))
    collisions = len(seeds) - distinct
    expected = len(seeds) ** 2 / 2 ** 32
    results.append({"check": "per_draw_seed_collisions", "value": collisions,
                    "bound": None, "verdict": "REPORTED",
                    "detail": f"{collisions} collision(s) among {len(seeds)} draws of one run; "
                              f"birthday expectation for a 31-bit space is {expected:.1f}. "
                              f"A collision is bit-identical noise only if the two draws also "
                              f"share an offset range, i.e. have the same element count."})

    failed = [r for r in results if r["verdict"] == "FAIL"]
    report = {
        "tool": "noise_stream_validation",
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "under_test": "kernels/rng_stream.py + kernels/ops/rand_op.py::randn_kernel",
        "n_per_draw": a.n, "draws": a.draws, "run_seed": a.seed,
        "sigma": SIGMA, "bounds": b,
        "results": results,
        "verdict": "PASS" if not failed else "FAIL",
        "failed": [r["check"] for r in failed],
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))

    width = max(len(r["check"]) for r in results)
    for r in results:
        bnd = "" if r["bound"] is None else f"  bound {r['bound']:.3e}"
        val = r["value"] if isinstance(r["value"], (int, bool)) else f"{r['value']:+.6e}"
        print(f"  {r['verdict']:8s} {r['check']:{width}s}  {val}{bnd}")
    print(f"\n{report['verdict']} — {len(results) - len(failed)}/{len(results)} checks within bound")
    print(f"written: {out}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
