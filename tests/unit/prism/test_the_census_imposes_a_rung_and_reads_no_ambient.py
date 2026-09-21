"""The certification census plans against an IMPOSED ladder rung, not the
machine's live free memory.

The census must be a function of the rung alone: the same model at the same
rung yields the same plan and the same keys on any machine and in any memory
weather (the tiling standard, owner 2026-09-22). Before this, `_prepare_devices`
sized a unified device against `memory_state().available_mb` — live host memory
that the census shadow does not neutralise — so the same model censused as
lazy_sequential one minute and layer_streaming the next: a coin toss, and two
CogVideoX defects that could not be reproduced.

`NBX_CENSUS_RUNG_MB` imposes the rung: capacity IS the rung, no ambient is read,
and the value is rounded onto the ladder so no budget is ever off-ladder.
"""
from __future__ import annotations

import pytest

from neurobrix.core.host_memory import MemoryState
from neurobrix.core.prism import solver as _solver_mod
from neurobrix.core.prism.loader import load_profile
from neurobrix.core.prism.solver import PrismSolver


def _solver():
    s = PrismSolver.__new__(PrismSolver)
    s.safety_margin = 0.95
    return s


def _cap(monkeypatch, *, available_mb, rung_env=None):
    """The unified device's planned capacity under a given host reading and
    an optional imposed rung."""
    monkeypatch.setattr(_solver_mod, "memory_state",
                        lambda: MemoryState(total_mb=24576,
                                            available_mb=available_mb,
                                            source="test"))
    if rung_env is None:
        monkeypatch.delenv("NBX_CENSUS_RUNG_MB", raising=False)
    else:
        monkeypatch.setenv("NBX_CENSUS_RUNG_MB", str(rung_env))
    profile = load_profile("default")            # apple, unified
    profile.devices[0].unified_memory = True
    devices = _solver()._prepare_devices(profile)
    return devices[0].capacity_mb


def test_without_a_rung_the_plan_still_reads_the_ambient():
    """The ordinary run is unchanged: a unified device sizes against what is
    actually free (this is the behaviour the census must NOT inherit)."""
    with pytest.MonkeyPatch.context() as mp:
        tight = _cap(mp, available_mb=6000)
        loose = _cap(mp, available_mb=20000)
    assert tight < loose, (tight, loose)   # ambient-dependent, as designed


def test_an_imposed_rung_is_the_capacity_and_ignores_the_ambient():
    with pytest.MonkeyPatch.context() as mp:
        cap = _cap(mp, available_mb=6000, rung_env=8192)
    assert cap == 8192, cap                # the rung, not min(recommended, 6000*0.95)


def test_the_same_rung_yields_the_same_capacity_in_any_memory_weather():
    """The whole point: the census stops being a coin toss."""
    with pytest.MonkeyPatch.context() as mp:
        starved = _cap(mp, available_mb=3000, rung_env=8192)
        idle = _cap(mp, available_mb=23000, rung_env=8192)
        unmeasured = _cap(mp, available_mb=None, rung_env=8192)
    assert starved == idle == unmeasured == 8192, (starved, idle, unmeasured)


def test_an_imposed_budget_is_rounded_onto_the_ladder():
    """No budget is ever an off-ladder value: 18000 MB rounds down to the
    16 GB rung."""
    with pytest.MonkeyPatch.context() as mp:
        cap = _cap(mp, available_mb=24000, rung_env=18000)
    assert cap == 16 * 1024, cap
