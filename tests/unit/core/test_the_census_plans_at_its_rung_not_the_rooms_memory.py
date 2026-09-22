"""A census shadow on UNIFIED memory plans at its rung, never at the room's free memory.

`_device_reading` already knows this — "a census shadow sees no card and carries the machine's
plan, not the room's" — but `_prepare_devices` lowers `capacity` to `host.available_mb` BEFORE
that reading is taken, on any unified device, with no shadow check:

    capacity = recommended
    if dev.has_unified_memory and host.measured:
        capacity = min(recommended, host.available_mb * self.safety_margin)

That lowering exists for a real RUN — a plan sized against the recommendation is accepted and
then killed mid-execution when the machine is busy (2026-09-10). A census executes nothing, so
the justification does not reach it, and the cost is severe: the tile of a tiled family is
derived from the budget, so its KEYS become a function of whatever else was running. Measured
2026-09-22 on this M4 Pro: the same model at the same imposed rung logged "planning against
8 659 MB actually free" in one shadow and "7 604 MB" in the next, minutes apart.

A census that is not reproducible is not a census.
"""
from __future__ import annotations

import pytest

from neurobrix.core.prism import solver as S


@pytest.fixture
def unified_profile():
    from neurobrix.core.prism.loader import load_profile
    p = load_profile("default-9f169c79")          # this machine: 1 x Apple M4 Pro, unified
    assert any(d.has_unified_memory for d in p.devices), "fixture needs a unified device"
    return p


def _capacities(profile):
    return [d.capacity_mb for d in S.PrismSolver()._prepare_devices(profile)]


def test_outside_a_census_a_busy_machine_still_lowers_the_plan(monkeypatch, unified_profile):
    """The 2026-09-10 repair must keep working for a real run."""
    monkeypatch.setattr(S, "memory_state", lambda: _Reading(measured=True, available_mb=3000.0))
    monkeypatch.setattr(S, "_census_shadow_active", lambda: False, raising=False)
    got = _capacities(unified_profile)
    assert max(got) < 6000, f"a busy machine must still lower a real plan: {got}"


def test_under_the_census_shadow_the_room_does_not_move_the_plan(monkeypatch, unified_profile):
    monkeypatch.setattr(S, "memory_state", lambda: _Reading(measured=True, available_mb=3000.0))
    monkeypatch.setattr(S, "_census_shadow_active", lambda: True, raising=False)
    got = _capacities(unified_profile)
    assert max(got) > 15000, (
        f"the shadow planned at the room's memory, not the profile's capacity: {got}")


class _Reading:
    def __init__(self, measured, available_mb):
        self.measured, self.available_mb = measured, available_mb
        self.source = "test"

    def describe(self):
        return "test reading"
