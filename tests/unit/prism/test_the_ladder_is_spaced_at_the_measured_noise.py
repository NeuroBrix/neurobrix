"""The ladder's spacing is DERIVED from the measured noise of a free-memory reading.

A free reading is not a constant, so a plan derived from one unrounded changes with the weather.
Measured on this rack 2026-09-24 — host `MemAvailable`, 120 samples over 60 s with four cards
busy, because a quiet machine understates exactly the swing the rounding defends against:

    median 218 785 MB   stdev 1 727 (0.79 %)   p5..p95 5 366 (2.45 %)   full swing 5 571 (2.55 %)

This session saw the consequence from the other side: the same Prism call returned
`single_gpu_lifecycle` during a 19-hour render holding ~19 GB of pinned host memory and
`single_gpu` afterwards, from byte-identical code (register 99). So the rounding stays. Only the
SPACING was ever in question.

WHY A HAND-PICKED LIST CANNOT BE RIGHT
--------------------------------------
The noise is RELATIVE — a fraction of the pool — and picked rungs are ABSOLUTE. Against the
measurement the old low range was 3.6x to 19.6x oversized:

    rung MB   step MB   noise MB   step/noise
       4096      2048        104        19.6
       8192      3072        209        14.7     a reading of 10 638 fell to 8 192
      16384      4096        418         9.8     a reading of 17 277 fell to 16 384

and the discarded memory is real: 2 206 MB at the first, 822 MB at the second. The second is
`granite-speech-3.3-8b`, whose largest component is 16 769.6 MB against a card reporting 17 277 —
the rung is what put it 386 MB out of reach and sent an 8 B model to the host.
"""
from __future__ import annotations

import pytest

from neurobrix.core.prism.memory_budget import rung_down_mb
from neurobrix.core.config.system import PRISM_DEFAULTS, _ladder_gb


def test_the_noise_is_declared_beside_the_ladder():
    n = PRISM_DEFAULTS["memory_reading_noise"]
    assert 0 < n < 1, n
    assert abs(n - 0.0255) < 1e-9, "the declared noise no longer matches the recorded measurement"


def test_no_rung_discards_more_than_the_measured_noise():
    """The whole property. Rounding down must cost at most the swing it absorbs — otherwise it
    is not absorbing noise, it is discarding memory."""
    # Through the function the ENGINE calls, never PRISM_DEFAULTS: reading the declared list
    # passed while memory_ladder_mb() truncated it to whole GB (vacuous-gates register 100).
    noise = PRISM_DEFAULTS["memory_reading_noise"]
    worst, at = 0.0, None
    for mb in range(4096, 512 * 1024, 337):        # 337: a prime stride, so readings are not all on rungs
        r = rung_down_mb(mb)
        if r <= 0:
            continue
        loss = (mb - r) / mb
        if loss > worst:
            worst, at = loss, mb
    assert worst <= noise + 1e-6, (
        f"a reading of {at} MB loses {worst*100:.2f} % to the rung below it, against a measured "
        f"noise of {noise*100:.2f} %. The ladder is discarding memory rather than absorbing a swing.")


def test_a_swing_of_the_measured_size_moves_the_rung_by_AT_MOST_ONE_step():
    """The guarantee is BOUNDED movement, not none — and my first version of this cell asked
    for none, which no finite ladder can give.

    A reading sitting just under a rung, plus a noise-sized swing, crosses into the next rung
    whatever the spacing: that is a boundary, not a defect. What the rounding must guarantee is
    that a swing of the measured size cannot carry a plan across MORE than one rung, so the
    budget moves by at most one step and never jumps.
    """
    from neurobrix.core.prism.memory_budget import memory_ladder_mb, rung_down_mb
    lad = memory_ladder_mb()
    noise = PRISM_DEFAULTS["memory_reading_noise"]
    idx = {v: i for i, v in enumerate(lad)}
    worst, at = 0, None
    for mb in range(4200, 512 * 1024, 911):            # 911: prime, so readings land off-rung
        lo, hi = rung_down_mb(mb), rung_down_mb(mb * (1 + noise))
        if lo not in idx or hi not in idx:
            continue
        moved = idx[hi] - idx[lo]
        if moved > worst:
            worst, at = moved, mb
    assert worst <= 1, (
        f"a reading of {at} MB plus one measured swing moves the budget {worst} rungs. A swing "
        f"must move it by at most one step, or the plan jumps rather than drifts.")


def test_no_step_is_so_fine_that_the_ladder_stops_absorbing_anything():
    """The opposite failure: steps far below the noise make the rounding decorative — every
    swing changes the rung and nothing is stabilised."""
    from neurobrix.core.prism.memory_budget import memory_ladder_mb
    lad = memory_ladder_mb()
    noise = PRISM_DEFAULTS["memory_reading_noise"]
    steps = [(b - a) / a for a, b in zip(lad, lad[1:])]
    median = sorted(steps)[len(steps) // 2]
    assert median >= noise * 0.5, (
        f"the median step is {median*100:.2f} % against a {noise*100:.2f} % swing; the ladder "
        f"is too fine to absorb the noise it was sized from")


def test_the_low_range_recovers_what_the_old_list_discarded():
    """The three readings this rack and the Mac actually measured."""
    # Through the function the ENGINE calls, never PRISM_DEFAULTS: reading the declared list
    # passed while memory_ladder_mb() truncated it to whole GB (vacuous-gates register 100).
    for reading, old_rung, floor in ((10638, 8192, 10000), (17277, 16384, 17000), (15565, 12288, 15000)):
        new = rung_down_mb(reading)
        assert new > old_rung, f"reading {reading}: new rung {new} is not above the old {old_rung}"
        assert new >= floor, f"reading {reading}: new rung {new} below the expected floor {floor}"


def test_granite_speechs_component_now_fits_the_rung_its_card_reports():
    """The case the ladder created: 16 769.6 MB of component, a card reporting 17 277, and a
    rung of 16 384 that put it 386 MB out of reach."""
    # Through the function the ENGINE calls, never PRISM_DEFAULTS: reading the declared list
    # passed while memory_ladder_mb() truncated it to whole GB (vacuous-gates register 100).
    assert rung_down_mb(17277) >= 16769.6, (
        "a 17 277 MB card still rounds below granite-speech's 16 769.6 MB component")


# ───────────────────────── the generator refuses nonsense ─────────────────────────

@pytest.mark.parametrize("bad", [0, 1, -0.1, 1.5])
def test_a_noise_that_is_not_a_fraction_is_REFUSED(bad):
    with pytest.raises(ValueError, match="not a fraction"):
        _ladder_gb(bad)


def test_the_ladder_spans_the_declared_range_and_is_monotonic():
    lad = _ladder_gb(0.0255, lo_gb=4, hi_gb=512)
    assert lad[0] == 4 and lad[-1] >= 512
    assert all(b > a for a, b in zip(lad, lad[1:]))
