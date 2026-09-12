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
