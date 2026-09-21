"""A plan that fits the card whole but not the memory actually free is not
accepted whole: the live reading, rounded onto the ladder, bounds every
whole-plan budget.

Measured 2026-09-20 on a V100-32GB: real-esrgan-x8 at 1024 needs 17 237 MB
whole; with a neighbour holding 18-27 GB the planner still wrote
`single_gpu, 17 237 MB planned` five times out of five — `free_mb` was the
plan's own accounting, the rung was consulted only on CAPACITY overflow —
and each run died at conv::349. The cells below feed the planner a reading
through the same door the engine reads, and ask the two things the law
requires: the reading is what the device state carries, and a whole plan
above the reading's rung is refused.
"""
from __future__ import annotations

import pytest

from neurobrix.core.prism.solver import (
    DeviceState, PrismSolver, memory_ladder_rung_mb,
)


def _solver():
    s = PrismSolver.__new__(PrismSolver)
    s.oom_reserve_mb = 3072
    return s


def test_the_effective_capacity_is_bounded_by_the_readings_rung():
    s = _solver()
    dev = DeviceState(device_string="cuda:0", capacity_mb=32000.0, used_mb=0.0)
    unbounded = s._effective_capacity_mb(dev)
    # nothing external holds the card: dedicated, used whole (the memory doctrine, 2026-09-21)
    assert unbounded == 32000.0
    # a neighbour holds 20 GB: the reading is 12 000 MB, its rung 12 288 > reading?
    # no — rounding DOWN: 12 000 sits on the 11 GB rung of the commercial ladder (11 264)
    dev.external_used_mb = 20000.0          # the machine's figure, not the plan's
    assert dev.free_mb == 12000.0
    assert s._effective_capacity_mb(dev) == 11 * 1024
    # 17 237 MB whole does not fit 11 264; it fitted 28 000-odd before
    assert 17237 > s._effective_capacity_mb(dev)
    assert 17237 <= unbounded


def test_prepare_devices_reads_the_driver_for_a_discrete_card(monkeypatch):
    from neurobrix.core.prism import solver as S
    from neurobrix.kernels.nbx_tensor import DeviceAllocator as D

    class _Dev:
        memory_mb = 32000
        has_unified_memory = False
        index = 0
        def get_device_string(self):
            return "cuda:0"

    class _Profile:
        devices = [_Dev()]

    monkeypatch.setattr(D, "free_memory_mb", staticmethod(lambda idx: 9000.0))
    # the sharing seam reads the same machine the driver reading describes: a neighbour holds 23 000 MB
    monkeypatch.setattr(S, "read_device_sharing", lambda idx: S.DeviceReading(
        kind="device", capacity_mb=32000.0, free_mb=9000.0, held_by_others_mb=23000.0, measured=True, source="test"))
    monkeypatch.setattr(S, "memory_state", lambda: type("M", (), {"measured": False, "available_mb": 0, "source": "test", "describe": lambda self: "test"})())
    s = _solver(); s.safety_margin = 1.0
    devs = s._prepare_devices(_Profile())
    assert devs[0].capacity_mb == 32000.0
    assert devs[0].free_mb == 9000.0, "the device state must carry the driver's reading"
    assert s._effective_capacity_mb(devs[0]) == 8 * 1024
