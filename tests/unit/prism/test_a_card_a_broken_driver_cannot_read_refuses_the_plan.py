"""A dedicated card the driver refuses to read refuses the plan: Prism never plans it as the profile's card,
unmeasured. Only a host with no GPU runtime at all (planning for a target profile, the machine elsewhere) plans the
profile's card, and the reading says so.

Before (qa, 2026-11-13): `_device_reading` caught every exception and returned the profile's capacity, unmeasured;
with nbxtensor raising DeviceQueryError for a driver that refuses (rc=700), that catch turned the refusal back into a
card of the profile's size.

What this test does if the code is wrong: a refusing driver yields a reading (red), or a host with no GPU runtime is
refused, or its reading does not say it is the profile's capacity unmeasured (red). No card."""
from __future__ import annotations

import pytest

from neurobrix.core.prism import solver as solver_mod

PROFILE = "c4140-4xv100-16GB-nvlink"


def _card():
    from neurobrix.core.prism.loader import load_profile
    prof = load_profile(PROFILE)
    return prof, prof.devices[1]


def test_a_driver_that_refuses_refuses_the_plan(monkeypatch):
    from nbxtensor.memory.allocator import DeviceQueryError

    def refused(_i):
        raise DeviceQueryError("cudaMemGetInfo failed with error 700 on device 1")
    monkeypatch.setattr(solver_mod, "read_device_sharing", refused)
    prof, dev = _card()
    with pytest.raises(DeviceQueryError, match="error 700"):
        solver_mod.PrismSolver()._device_reading(dev, float(dev.memory_mb), None, prof)


def test_a_host_with_no_gpu_runtime_plans_the_profiles_card_and_says_so(monkeypatch):
    from nbxtensor.backend.runtime import NoGpuRuntime

    def absent(_i):
        raise NoGpuRuntime("No GPU runtime found")
    monkeypatch.setattr(solver_mod, "read_device_sharing", absent)
    prof, dev = _card()
    r = solver_mod.PrismSolver()._device_reading(dev, float(dev.memory_mb), None, prof)
    assert r.capacity_mb == float(dev.memory_mb) and not r.measured
    assert r.source == "no GPU runtime on this host: the profile's capacity, unmeasured"
