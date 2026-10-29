"""A whole plan is budgeted against the pool's declared fact; what is free at the moment is the DOOR at
execution (the owner, 2026-10-25 15:16), never a smaller budget. Before: a neighbour holding 20 GB of a V100-32GB
lowered the budget to the 11 GB rung and the plan moved; now the plan stays and the run refuses by name when the
neighbour is still there. The cells feed the planner a reading through the same door the engine reads.
"""
from __future__ import annotations
import pytest
from neurobrix.core.prism.solver import DeviceState, PrismSolver, memory_ladder_rung_mb
from neurobrix.core.prism.memory_budget import DeviceReading, execution_door


def _solver():
    s = PrismSolver.__new__(PrismSolver)
    s.oom_reserve_mb = 3072
    return s


def test_the_effective_capacity_is_the_pools_fact_whatever_a_neighbour_holds():
    s = _solver()
    dev = DeviceState(device_string="cuda:0", capacity_mb=32000.0, used_mb=0.0)
    assert s._effective_capacity_mb(dev) == 32000.0
    dev.external_used_mb = 20000.0          # the machine's figure, not the plan's
    assert dev.free_mb == 12000.0
    assert s._effective_capacity_mb(dev) == 32000.0            # the budget does not move
    door = execution_door(DeviceReading(kind="device", capacity_mb=32000.0, free_mb=12000.0, held_by_others_mb=20000.0), 17237)
    assert door and "20000 MB held by other processes" in door   # the run refuses by name instead


def test_prepare_devices_reads_the_driver_for_a_discrete_card(monkeypatch):
    from neurobrix.core.prism import solver as S
    from nbxtensor.memory.allocator import DeviceAllocator as D
    class _Dev:
        memory_mb = 32000
        has_unified_memory = False
        index = 0
        def get_device_string(self):
            return "cuda:0"
    class _Profile:
        devices = [_Dev()]
    monkeypatch.setattr(D, "free_memory_mb", staticmethod(lambda idx: 9000.0))
    monkeypatch.setattr(S, "read_device_sharing", lambda idx: S.DeviceReading(
        kind="device", capacity_mb=32000.0, free_mb=9000.0, held_by_others_mb=23000.0, measured=True, source="test"))
    monkeypatch.setattr(S, "memory_state", lambda: type("M", (), {"measured": False, "available_mb": 0, "source": "test", "describe": lambda self: "test"})())
    monkeypatch.setattr(S, "_census_shadow_active", lambda: False)
    s = _solver()
    devs = s._prepare_devices(_Profile())
    assert devs[0].capacity_mb == 32000.0
    assert devs[0].free_mb == 32000.0, "what others hold is the door's business, not the plan's room (2026-10-25)"
    assert devs[0].live_free_mb == 9000.0, "the door carries the moment"
    assert s._effective_capacity_mb(devs[0]) == 32000.0, "the budget is the card's fact"
