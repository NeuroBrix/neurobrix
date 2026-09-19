"""A budget whose variance exceeds its own shortfall cannot decide anything.

`effective_capacity = capacity_mb - oom_reserve_mb` held back a flat 3072 MB
whatever the card. The reading that budget is taken against is not flat: measured
here at a FIXED point in a FIXED computation -- the same op of the same request,
refused three times -- the driver said free

    live_tracked=8242MB    5662 / 6586 / 7598 MB    spread 1936 MB = 12.0% of 16151
    live_tracked=10073MB   1390 / 2042 / 2350 MB    spread  960 MB =  5.9%

and the Mac saw 12598 -> 15248 MB on an idle machine, 21%. Its `real-esrgan-x8`
refusal missed by 27 MB, two orders below that spread: the plan's verdict there
was noise wearing a number.

AN ABSOLUTE RESERVE AGAINST A PROPORTIONAL VARIANCE is adequate at one card size
and short at another, and this rack holds both:

    16151 MB card:  3072 MB = 19.0%   over the 12.0% spread   -> unchanged
    32501 MB card:  3072 MB =  9.5%   UNDER it, short by 828  -> the margin moves

So the margin is the larger of the flat reserve and the measured spread, and the
plan STATES it. Stating it is half the point: a margin a reader has to discover
by reading the solver is the same silence as no margin.

SEEN RED: with `_margin_mb` returning `int(self.oom_reserve_mb)`,
`test_the_margin_covers_the_measured_spread_on_a_large_card` fails; with the
`margin` line removed from `explain_plan`, `test_the_plan_says_what_it_held_back`
fails.

Run: PYTHONPATH=src python -m pytest tests/unit/prism/test_the_plan_states_the_margin_it_keeps.py
"""
from __future__ import annotations

import pytest

from neurobrix.core.prism.solver import (
    ComponentAllocation, ComponentMemory, ExecutionPlan, PrismSolver, explain_plan,
)

SMALL = 16151      # this rack's 16 GB V100s, as the driver reports them
LARGE = 32501      # and its 32 GB ones


def _solver():
    """A solver without __init__: these cells ask about one method only.

    The volatility lives on the CLASS for exactly this reason -- an earlier
    version set it in __init__ and five cells that build a solver this way went
    red with AttributeError, which says nothing about the margin.
    """
    s = PrismSolver.__new__(PrismSolver)
    s.oom_reserve_mb = 3072
    return s


def test_the_flat_reserve_still_wins_on_a_small_card():
    """12% of 16151 is 1938 MB, under the 3072 MB reserve: nothing moves.

    Pinned because the change must NOT quietly loosen the small cards, which
    are where every refusal tonight happened.
    """
    assert _solver()._margin_mb(SMALL) == 3072


def test_the_margin_covers_the_measured_spread_on_a_large_card():
    """12% of 32501 is 3900 MB, over the reserve: the margin follows the card."""
    m = _solver()._margin_mb(LARGE)
    assert m == int(LARGE * 0.12) == 3900, m
    assert m > 3072, "an absolute reserve is short of a proportional variance here"


def test_the_margin_is_never_below_the_flat_reserve():
    """A tiny card must not end up with a tinier margin than the reserve."""
    for cap in (2048, 8192, SMALL, LARGE, 81559):
        assert _solver()._margin_mb(cap) >= 3072, cap


def test_the_volatility_is_the_measured_one_and_survives_a_bare_solver():
    assert PrismSolver.reading_volatility == 0.12
    assert _solver().reading_volatility == 0.12


def test_the_plan_says_what_it_held_back():
    """A margin a reader must find in the solver is the same as no margin."""
    comps = {"model": ComponentAllocation(name="model", devices=["cuda:0"],
                                          dtype="float16", memory_mb=32,
                                          architecture="x", vendor="y")}
    mem = {"model": ComponentMemory("model", 32 * 2**20, 2**30, 2**26)}
    plan = ExecutionPlan(components=comps, target_dtype="float16",
                         total_memory_mb=32, strategy="single_gpu",
                         component_memory=mem, selection_reason="r",
                         candidates=[("single_gpu", 1000.0)], rejected=[],
                         margin_mb=3900,
                         margin_reason="12% of 32501 MB, the measured spread of "
                                       "this driver's own free-memory reading")
    text = explain_plan(plan)
    assert "margin" in text and "3900 MB held back per card" in text
    assert "measured spread" in text


def test_a_plan_without_a_margin_invents_no_line():
    """Otherwise the test above passes against a hardcoded string."""
    comps = {"model": ComponentAllocation(name="model", devices=["cuda:0"],
                                          dtype="float16", memory_mb=32,
                                          architecture="x", vendor="y")}
    mem = {"model": ComponentMemory("model", 32 * 2**20, 2**30, 2**26)}
    plan = ExecutionPlan(components=comps, target_dtype="float16",
                         total_memory_mb=32, strategy="single_gpu",
                         component_memory=mem, selection_reason="r",
                         candidates=[("single_gpu", 1000.0)], rejected=[])
    assert "held back per card" not in explain_plan(plan)
