"""A hardware profile plans ITS card, whatever card it runs on (rule 1: the profile is the machine's identity card).
A 16 GB V100 profile run on a 32 GB V100 read the driver's 32 510 MB and budgeted 16 384 MB, where every 16 GB card gives
15 845 MB (its measured runtime total 16 151 less the runtime context 306): Janus-Pro-7B native planned its language
model whole (129 s estimated) on the 32 GB card and streamed it on the 16 GB one (1 062 s), the same profile, the same
tree (explain-plan, 2026-11-02 18:50). The reading is now capped at the profile's measured card; a smaller or busier card
still lowers it.

What this test would do if the code were wrong: the bigger card's budget, capacity or free figure above the profile's
card (RED), or the profile's card lowered when the real card is the same (RED). No card."""
from __future__ import annotations

import copy

from neurobrix.core.prism import solver as S
from tests.unit.prism._pinned_machine import V100_16GB, pin_dedicated_card, profile

MEASURED = {"runtime_total_mb": 16151, "runtime_context_mb": 306}     # the rack's default-ff6008b7, measured 2026-10-27


def _reading(monkeypatch, driver_total_mb):
    spec = copy.deepcopy(V100_16GB)
    spec["id"] = "scenario-v100-16gb-measured"
    spec["devices"][0].update(MEASURED)
    pin_dedicated_card(monkeypatch, driver_total_mb, MEASURED["runtime_context_mb"], f"a {driver_total_mb} MB card")
    prof = profile(spec)
    solver = S.PrismSolver()
    dev = prof.devices[0]
    r = solver._device_reading(dev, float(dev.memory_mb), host=None, profile=prof)
    return r, S._capped_budget_mb(r, float(dev.memory_mb))


def test_the_16gb_profile_reads_a_32gb_card_as_its_own():
    import pytest
    mp = pytest.MonkeyPatch()
    try:
        r16, b16 = _reading(mp, 16151)
        r32, b32 = _reading(mp, 32510)
    finally:
        mp.undo()
    assert (r32.capacity_mb, r32.free_mb) == (r16.capacity_mb, r16.free_mb) == (16151.0, 15845.0)
    assert b32 == b16 == 15845.0


def test_a_smaller_card_still_lowers_the_reading(monkeypatch):
    r, b = _reading(monkeypatch, 15000)
    assert r.capacity_mb == 15000.0 and b == 15000.0 - MEASURED["runtime_context_mb"]
