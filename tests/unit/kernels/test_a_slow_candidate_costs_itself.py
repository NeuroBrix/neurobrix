"""A pathologically slow candidate costs itself, named with both its numbers.

The third species, measured 2026-09-12 on a bf16 matmul sweep at M=1500:
neither refused nor wrong, but one launch of ~2.5 minutes -- which the
five-launch estimate multiplied past twelve. Neither the refusal exclusion nor
the oracle sees it; only the campaign timeout reaped it, killing the run
without naming the candidate.

Three rules, and each has its test:

  * the budget is DERIVED from the sweep's own measurements (the oracle's CPU
    time for the first candidate, the fastest completed probe for the rest) —
    never a constant. Until the ratios are measured, they are None and the
    watchdog is DISARMED: it must never guess;
  * an over-budget candidate scores `inf` and the sweep continues;
  * the exclusion is SAID with both numbers — a candidate scored out for time
    without its time is a silent narrowing.

The GPU test that sees the budget bite a manufactured hanger lives beside the
ratio measurement that arms it; these pin the machinery both ways first.

Runnable: PYTHONPATH=src python3 -m pytest \
    tests/unit/kernels/test_a_slow_candidate_costs_itself.py -v
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _fresh_sweep():
    from neurobrix.kernels import autotune_refusals as R

    saved = dict(R._BUDGET_RATIOS)
    R.begin_sweep()
    yield R
    R._BUDGET_RATIOS.update(saved)
    R.begin_sweep()


def test_disarmed_until_its_ratios_are_measured(_fresh_sweep):
    R = _fresh_sweep
    R._BUDGET_RATIOS.update(first_vs_oracle_cpu=None, later_vs_best=None)
    R.note_candidate_time(3.0)
    assert R._compute_budget() is None, (
        "an unmeasured ratio must disarm the watchdog entirely: a guessed "
        "budget is a constant wearing a derivation's clothes")


def test_the_budget_derives_from_the_fastest_completed_probe(_fresh_sweep):
    R = _fresh_sweep
    R._BUDGET_RATIOS.update(first_vs_oracle_cpu=None, later_vs_best=10.0)
    R.note_candidate_time(4.0)
    R.note_candidate_time(2.0)          # the winner sets the scale
    R.note_candidate_time(6.0)
    assert R._compute_budget() == pytest.approx(20.0), (
        "later candidates are budgeted against the BEST completed probe, not "
        "the last or the worst")


def test_an_over_budget_candidate_scores_inf_and_is_said(_fresh_sweep):
    from neurobrix.kernels.launcher import CandidateOverTimeBudget

    R = _fresh_sweep
    said = []

    def bench():
        raise CandidateOverTimeBudget(150000.0, 20.0)

    out = R.exclude_slow_candidates(bench, say=said.append)()
    assert out == [float("inf")] * 3
    assert len(said) == 1
    assert "150000.0" in said[0] and "20.0" in said[0], (
        f"both numbers must be in the announcement; got {said[0]!r}")


def test_anything_else_still_propagates(_fresh_sweep):
    """The same non-negotiable as the refusal exclusion: only the dedicated
    exception is caught. An OOM or a kernel bug ending up as `inf` would turn
    every failure into a silently slower shape."""
    R = _fresh_sweep

    def bench():
        raise ValueError("not a budget matter")

    with pytest.raises(ValueError):
        R.exclude_slow_candidates(bench)()


def test_a_new_sweep_forgets_the_old_scale(_fresh_sweep):
    R = _fresh_sweep
    R._BUDGET_RATIOS.update(later_vs_best=10.0)
    R.note_candidate_time(2.0)
    assert R._compute_budget() == pytest.approx(20.0)
    R.begin_sweep()
    assert R._SWEEP["best_ms"] is None, (
        "a key's scale must not leak into the next key's budget: a tiny "
        "kernel's 0.1 ms would sentence every candidate of a large one")
