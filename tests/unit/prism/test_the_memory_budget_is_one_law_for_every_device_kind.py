"""Hocine's memory doctrine, as one law in `core/prism/memory_budget.py` (the owner, 2026-09-21):

* a SHARED pool — unified memory, a display on the device, memory another process holds, a
  reading that could not be taken, host RAM — is budgeted at its FREE reading rounded DOWN onto
  the commercial ladder, never a value off the ladder;
* a DEDICATED card nothing else uses is used whole, less only the runtime's own context;
* the tile a plan cuts derives from the RUNG: a dedicated card's nominal rung, a shared pool's
  free rung.

What main did before (measured): an idle V100-16GB reading 15 700 MB free was budgeted at
12 288 MB (the ladder of the free reading, then a margin of max(3 072 MB, 12 %) on top), an idle
V100-32GB at 24 576 MB; a reading under 4 096 MB passed through the ladder untouched; host RAM
was 0.7 × the installed figure and never met the ladder.

Shapes: 16 384 and 32 768 are this rack's two card sizes, 15 700 and 31 700 what an idle card
of each reads free (its own context taken), 1 396 MB what one certification round holds on a
neighbour's card at the time of writing; 18 186 MB is what the Mac's M4 Pro recommends and
10 099 MB what it had free when a plan sized on the recommendation was killed (2026-09-10);
3 000 MB sits under the lowest rung; 200 000 MB of host RAM sits between the 192 and 256 rungs.
"""
from __future__ import annotations

import pytest

from neurobrix.core.prism.memory_budget import (DeviceReading, budget_mb, describe, host_budget_mb,
                                                is_shared, memory_ladder_mb, rung_down_mb, tile_rung_mb)


def test_the_ladder_is_configuration_from_four_to_five_hundred_twelve_gigabytes():
    rungs = memory_ladder_mb()
    assert rungs[0] == 4 * 1024 and rungs[-1] == 512 * 1024, rungs
    assert 192 * 1024 in rungs and 256 * 1024 in rungs, rungs
    assert rungs == sorted(rungs)


@pytest.mark.parametrize("reading, rung", [(15700, 12 * 1024), (16384, 16 * 1024), (3000, 0), (200000, 192 * 1024),
                                           (31700, 24 * 1024), (600 * 1024, 512 * 1024)])
def test_a_reading_rounds_down_onto_the_ladder_and_never_passes_through(reading, rung):
    assert rung_down_mb(reading) == rung


def _idle(capacity, free, context=400.0):
    return DeviceReading(kind="device", capacity_mb=capacity, free_mb=free, own_context_mb=context, measured=True)


def test_an_idle_dedicated_card_is_used_whole_less_its_own_context():
    for cap, free in ((16384, 15700), (32768, 31700)):
        r = _idle(cap, free)
        assert not is_shared(r)
        assert budget_mb(r) == cap - 400, describe(r)
        assert tile_rung_mb(r) == cap, describe(r)         # the card's nominal rung sizes the tile


def test_a_dedicated_card_partly_held_by_another_process_is_the_remainder_rounded_down():
    r = DeviceReading(kind="device", capacity_mb=32768, free_mb=31700 - 1396, held_by_others_mb=1396, own_context_mb=400)
    assert is_shared(r)
    assert budget_mb(r) == 24 * 1024, describe(r)
    assert tile_rung_mb(r) == 24 * 1024


def test_a_display_makes_a_card_shared_whatever_it_holds():
    r = DeviceReading(kind="device", capacity_mb=16384, free_mb=15700, display_active=True)
    assert is_shared(r) and budget_mb(r) == 12 * 1024, describe(r)


def test_unified_memory_is_the_hosts_free_reading_rounded_down():
    r = DeviceReading(kind="device", capacity_mb=18186, free_mb=10099, unified=True)
    assert is_shared(r) and budget_mb(r) == 8 * 1024 and tile_rung_mb(r) == 8 * 1024, describe(r)


def test_a_reading_that_could_not_be_taken_is_the_shared_side():
    r = DeviceReading(kind="device", capacity_mb=16384, free_mb=15700, measured=False)
    assert is_shared(r) and budget_mb(r) == 12 * 1024


def test_host_ram_goes_through_the_same_ladder():
    r = DeviceReading(kind="host", capacity_mb=257000, free_mb=200000)
    assert is_shared(r) and budget_mb(r) == 192 * 1024
    assert host_budget_mb() in memory_ladder_mb() + [0]      # the machine's own reading lands on a rung


def test_the_solver_budgets_a_hand_built_device_state_by_the_same_law():
    from neurobrix.core.prism.solver import DeviceState, PrismSolver
    s = PrismSolver.__new__(PrismSolver)
    idle = DeviceState(device_string="cuda:0", capacity_mb=16384.0)
    assert s._effective_capacity_mb(idle) == 16384.0            # dedicated: whole (a state carries no context figure)
    held = DeviceState(device_string="cuda:1", capacity_mb=32768.0, external_used_mb=1396.0)
    assert s._effective_capacity_mb(held) == 24 * 1024          # shared: free 31 372 rounds down


def test_the_door_names_the_rung_for_a_census(monkeypatch):
    monkeypatch.setenv("NBX_PRISM_BUDGET_MB", "8192")
    r = _idle(32768, 31700)
    assert budget_mb(r) == 8192 and tile_rung_mb(r) == 8192 and "door" in describe(r)
