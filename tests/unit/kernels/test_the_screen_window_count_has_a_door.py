"""`_SCREEN_WINDOWS` was a constant with no door, and the instruction was to tune it.

The windowed screen replaced a fixed 1 GiB screening budget that left the largest shapes the
least verified. How many row windows it takes is the knob that trades cost against coverage —
and it was a module-level `3` that no measurement could reach.

WHAT THE COST ACTUALLY IS, measured on this rack 2026-09-23
-----------------------------------------------------------
chatterbox on a 16 GB V100, where its 32 GB-certified keys do not serve, so all 65 keys sweep
and the screen runs on every one:

    oracle OFF   379.9, 378.9 s    spread   1.0 s
    oracle ON    500.3, 395.5 s    spread 104.8 s

Every ON run is slower than every OFF run — 1.04x at the closest bound, 1.32x at the widest,
1.18x on the means. But the ON arm's own spread is 105 s against the OFF arm's 1 s, so the
variance lives in the oracle path and **a point factor from two reps is not a number**. The
first version of this measurement reported 1.32x from one rep per arm; the second rep of the
same arm came back 105 s faster and withdrew it.

The Apple side measured 9.98 s -> 313 s, **31.4x**, for ONE selection on a 4.6 GB shape. Two
machines an order of magnitude apart means the factor scales with what is windowed, so there is
no constant to adopt — which is exactly why the knob needs a door rather than a better default.
"""
from __future__ import annotations

import importlib

import pytest


def _launcher():
    m = importlib.import_module("neurobrix.kernels.launcher")
    return m


def test_the_default_is_unchanged():
    assert _launcher()._screen_windows() == 3


def test_the_door_is_read_when_used_not_frozen_at_import(monkeypatch):
    """A value frozen from the environment at import is the same defect as a literal standing
    in for a runtime value. `autotune_cache._dir` was bitten by exactly that and says so."""
    m = _launcher()
    monkeypatch.setenv("NBX_SCREEN_WINDOWS", "5")
    assert m._screen_windows() == 5
    monkeypatch.setenv("NBX_SCREEN_WINDOWS", "1")
    assert m._screen_windows() == 1


@pytest.mark.parametrize("bad", ["0", "-2"])
def test_a_count_that_would_screen_nothing_is_REFUSED(monkeypatch, bad):
    monkeypatch.setenv("NBX_SCREEN_WINDOWS", bad)
    with pytest.raises(ValueError, match="minimum"):
        _launcher()._screen_windows()


def test_a_non_integer_is_REFUSED_not_silently_defaulted(monkeypatch):
    """ZERO FALLBACK: a typo that silently returns 3 makes a whole sweep unattributable."""
    monkeypatch.setenv("NBX_SCREEN_WINDOWS", "three")
    with pytest.raises(ValueError, match="not an integer"):
        _launcher()._screen_windows()


# ───────────────────── the windows the door produces ─────────────────────

class _Out:
    def __init__(self, m, n, itemsize=2):
        self.shape = (m, n)
        self._nbytes = m * n * itemsize


@pytest.mark.parametrize("n_win", [1, 2, 3, 5])
def test_the_LAST_window_is_always_anchored_at_the_final_row(monkeypatch, n_win):
    """The failure class that motivates screening a large shape is index overflow, and it shows
    at the largest linear index or nowhere. Whatever the count, the last row must be covered."""
    monkeypatch.setenv("NBX_SCREEN_WINDOWS", str(n_win))
    m = _launcher()
    wins = m._row_windows_for(_Out(100_000, 64), budget_bytes=4 * 1024 * 1024)
    assert wins, "no windows produced"
    assert wins[-1][1] == 100_000, f"the last window does not reach the final row: {wins[-1]}"


@pytest.mark.parametrize("n_win", [1, 2, 3, 5])
def test_the_count_is_honoured_and_the_windows_do_not_overlap(monkeypatch, n_win):
    monkeypatch.setenv("NBX_SCREEN_WINDOWS", str(n_win))
    m = _launcher()
    wins = m._row_windows_for(_Out(100_000, 64), budget_bytes=4 * 1024 * 1024)
    assert len(wins) == n_win, f"asked for {n_win} windows, got {len(wins)}: {wins}"
    for (a0, a1), (b0, b1) in zip(wins, wins[1:]):
        assert a1 <= b0, f"windows overlap: {(a0,a1)} then {(b0,b1)}"


def test_the_count_REDISTRIBUTES_a_fixed_cost_it_does_not_raise_it(monkeypatch):
    """The window count is NOT a cost knob, and this cell exists because I assumed it was.

    `per = budget // (n_win * oracle_row_bytes)`, so more windows means proportionally smaller
    windows and the TOTAL rows the oracle computes stays put. Measured, M=100 000, N=64,
    4 MiB budget:

        windows   rows each   total rows   places
              1        8192         8192        1
              2        4096         8192        2
              3        2730         8190        3
              5        1638         8190        5
              8        1024         8192        8

    So tuning this knob changes WHERE the screen looks, not what it costs. The cost is set by
    the screening budget. The first version of this cell asserted `rows[1] < rows[3] < rows[5]`
    and went red on its first run, which is how the property was found — the owner's
    instruction was to tune the count from a cost measurement, and the count is not what the
    cost responds to.
    """
    m = _launcher()
    totals, places = {}, {}
    for n in (1, 2, 3, 5, 8):
        monkeypatch.setenv("NBX_SCREEN_WINDOWS", str(n))
        wins = m._row_windows_for(_Out(100_000, 64), budget_bytes=4 * 1024 * 1024)
        totals[n] = sum(r1 - r0 for r0, r1 in wins)
        places[n] = len(wins)
    spread = max(totals.values()) - min(totals.values())
    assert spread <= max(totals.values()) * 0.01, (
        f"the window count changed the oracle's total work by {spread} rows; it is supposed to "
        f"redistribute a fixed budget, not raise it: {totals}")
    assert places == {1: 1, 2: 2, 3: 3, 5: 5, 8: 8}, places


def test_the_SCREENING_BUDGET_is_what_moves_the_cost(monkeypatch):
    """The knob that does trade cost for coverage, pinned beside the one that does not."""
    monkeypatch.setenv("NBX_SCREEN_WINDOWS", "3")
    m = _launcher()
    small = sum(b - a for a, b in m._row_windows_for(_Out(100_000, 64), budget_bytes=1 * 1024 * 1024))
    large = sum(b - a for a, b in m._row_windows_for(_Out(100_000, 64), budget_bytes=8 * 1024 * 1024))
    assert large > small * 3, f"the budget barely moved the work: {small} -> {large}"


def test_a_shape_that_cannot_be_windowed_says_so(monkeypatch):
    """None means 'I did not screen this', and the caller prints that rather than pretending.

    Until 2026-09-26 a 3-D output was such a shape and every convolution key over the budget
    (a 4-D `[N, Cout, Ho, Wo]`) was seated without a screen. An N-D contiguous output is rows
    of its last dimension — `[prod(leading), last]` — and windows like a matrix; what cannot be
    windowed by rows is an output with no rows at all."""
    monkeypatch.setenv("NBX_SCREEN_WINDOWS", "3")
    m = _launcher()

    class _OneD:
        shape = (256,)
        _nbytes = 256 * 2
    assert m._row_windows_for(_OneD(), budget_bytes=1024) is None

    class _FourD:                                      # a conv output: 1 x 3 x 2048 x 1024, fp16
        shape = (1, 3, 2048, 1024)
        _nbytes = 3 * 2048 * 1024 * 2
    wins = m._row_windows_for(_FourD(), budget_bytes=32 * 1024 * 1024)
    assert wins and wins[-1][1] == 3 * 2048, f"the flat rows are N*Cout*Ho = 6144: {wins}"
    assert all(r1 - r0 <= 32 * 1024 * 1024 // (3 * 1024 * 8) for r0, r1 in wins), wins


def test_the_oracle_row_cost_shrinks_the_window_when_the_kernel_says_so(monkeypatch):
    """A 3-channel output row is 24 KB of float64; its 128-channel receptive field is 2 MB. The
    convolution family sizes its windows by the latter, or a window of a few hundred rows reads
    gigabytes (the Mac's key, 2026-09-25)."""
    monkeypatch.setenv("NBX_SCREEN_WINDOWS", "3")
    m = _launcher()

    class _FourD:
        shape = (1, 3, 2048, 1024)
        _nbytes = 3 * 2048 * 1024 * 2
    budget = 32 * 1024 * 1024
    by_row = m._row_windows_for(_FourD(), budget_bytes=budget)
    by_field = m._row_windows_for(_FourD(), budget_bytes=budget, oracle_row_cost=2_121_728)
    assert by_field and by_row
    assert max(r1 - r0 for r0, r1 in by_field) < max(r1 - r0 for r0, r1 in by_row), (by_field, by_row)
    assert max(r1 - r0 for r0, r1 in by_field) * 2_121_728 * 3 <= budget, by_field
