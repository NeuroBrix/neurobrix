"""A sweep whose arguments exceed available memory is cut, said, not persisted.

Observed live 2026-09-13: a baddbmm key carrying 5.9 GB of arguments while the
machine had 4.5 GB available. Past the SCREEN budget the compare is skipped --
but Triton still times every candidate on those arguments, and under unified
memory that sweep measures the swap, not the kernels.

Three rules, each pinned:

  * the decision is CONSTANT-FREE: arguments alone exceeding `available_mb`
    (from `core.host_memory`, our own authority, read live) is the whole test;
  * the sweep is cut to the single first-declared config and the cut is SAID
    with all three numbers;
  * the choice is marked UNMEASURED and `capture()` never persists it --
    recorded, it would outlive the pressure that forced it and keep deciding
    on machines and days it knows nothing about.

Kept apart from the round-9 hang on purpose: that sweep's arguments were
~11 MB, so this door does not claim to explain it. The hang stays unattributed
until it reproduces; this door stands on its own observation.

Runnable: PYTHONPATH=src python3 -m pytest \
    tests/unit/kernels/test_bench_refuses_to_measure_the_swap.py -v
"""
from __future__ import annotations

import pytest


def test_the_decision_is_the_live_available_memory(monkeypatch):
    from neurobrix.kernels import launcher as L
    import neurobrix.core.host_memory as hm

    class _M:
        available_mb = 100
        total_mb = 24576
        source = "test"

    monkeypatch.setattr(hm, "memory_state", lambda: _M())
    swaps, avail = L.bench_would_swap(101 * 2 ** 20)
    assert swaps is True and avail == 100
    swaps, _ = L.bench_would_swap(99 * 2 ** 20)
    assert swaps is False, (
        "under the line nothing is gated: the comparison is the whole test, "
        "with no margin constant to tune")


def test_an_unreadable_platform_gates_nothing(monkeypatch):
    """`memory_state` names why it cannot read; this door must not turn that
    honest None into a silent always-on or always-off policy of its own."""
    from neurobrix.kernels import launcher as L
    import neurobrix.core.host_memory as hm

    class _M:
        available_mb = None
        total_mb = None
        source = "unreadable platform"

    monkeypatch.setattr(hm, "memory_state", lambda: _M())
    swaps, avail = L.bench_would_swap(10 ** 12)
    assert swaps is False and avail is None


def test_an_unmeasured_choice_is_never_persisted():
    from neurobrix.triton import autotune_cache as atc

    class _At:
        pass

    at = _At()
    key = ("k", 1)
    assert atc.is_unmeasured(at, key) is False, (
        "both directions: a key nobody marked must not be swallowed")
    atc.mark_unmeasured(at, key)
    assert atc.is_unmeasured(at, key) is True
    src = open(atc.__file__.replace(".pyc", ".py")).read()
    assert "is_unmeasured(at, key)" in src.split("def capture")[1], (
        "capture() must consult the registry; a registry nothing consults is "
        "the vacuous form")


# ── the post-hoc half: a sweep that GREW the swap measured the swap ────────
#
# The pre-gate above catches the certain case (arguments alone exceed
# available). Reproduced 2026-09-13, it is not enough: a 5.9 GB sweep PASSED
# at 6.5 GB available, its buffers saturated the machine, and the FOLLOWING
# work crawled -- stack sampled in waitUntilCompleted, swap at 7621/8192 MB.
# Round 9's hang had the same shape: its own arguments were 11 MB and the
# pressure was inherited from earlier keys.
#
# The threshold is ZERO, and zero is the identity and not a tuned constant.
# The costs are asymmetric: a false mark (background daemon moved the swap)
# only skips persistence and the key re-sweeps another day; a missed mark
# persists a choice timed against the swap, which then keeps deciding on
# machines and days it knows nothing about.


def test_a_sweep_that_grew_the_swap_is_marked_unmeasured(monkeypatch):
    from neurobrix.kernels import autotune_refusals as R
    import neurobrix.core.host_memory as hm

    swap = {"used": 4000}

    class _M:
        available_mb = 9000
        total_mb = 24576
        source = "test"

        @property
        def swap_used_mb(self):
            return swap["used"]

    monkeypatch.setattr(hm, "memory_state", lambda: _M())
    said = []

    class _At:
        pass

    at = _At()
    R.begin_sweep()
    R.note_sweep_swap_baseline()
    swap["used"] = 4500                    # the sweep grew the swap
    grew = R.sweep_grew_the_swap()
    assert grew == 500
    R.mark_sweep_unmeasured(at, ("k", 1), grew, say=said.append)
    from neurobrix.triton import autotune_cache as atc

    assert atc.is_unmeasured(at, ("k", 1)) is True
    assert len(said) == 1 and "500" in said[0], (
        f"the mark is SAID with its delta; got {said!r}")


def test_a_quiet_sweep_is_not_marked(monkeypatch):
    """Both directions: zero growth marks nothing, or every sweep on a busy
    machine would starve the persistent cache for no reason."""
    from neurobrix.kernels import autotune_refusals as R
    import neurobrix.core.host_memory as hm

    class _M:
        available_mb = 9000
        total_mb = 24576
        source = "test"
        swap_used_mb = 4000

    monkeypatch.setattr(hm, "memory_state", lambda: _M())
    R.begin_sweep()
    R.note_sweep_swap_baseline()
    assert R.sweep_grew_the_swap() == 0


def test_an_unreadable_swap_marks_nothing(monkeypatch):
    from neurobrix.kernels import autotune_refusals as R
    import neurobrix.core.host_memory as hm

    class _M:
        available_mb = None
        total_mb = None
        source = "unreadable"
        swap_used_mb = None

    monkeypatch.setattr(hm, "memory_state", lambda: _M())
    R.begin_sweep()
    R.note_sweep_swap_baseline()
    assert R.sweep_grew_the_swap() == 0, (
        "an unreadable platform must not turn into a policy of its own")
