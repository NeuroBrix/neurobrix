"""The Apple hardware profile publishes the largest single allocation its device grants.

A Metal device holds `memory_mb` in all (`recommendedMaxWorkingSetSize`) and refuses any ONE buffer above
`MTLDevice.maxBufferLength`, however much is free. On the M4 Pro: 13 639 MB against 18 186 MB. An arena is one
buffer. Measured 2026-10-04: at 17.1 GB free Prism cut deepseek-moe-16b-chat in 3 segments, the first arena
asked 14 661 558 272 bytes and the device refused it ("GPU malloc failed (error 1) for 14661558272 bytes [...
driver_free=17785MB / driver_total=18186MB]"); 4 segments of ~10 GB allocate. The profile did not carry the fact,
so no plan could respect it.

The detection asks the device (never a table) and the profile's device carries `max_allocation_mb`; a profile
that is silent says "no limit below the device's memory", which is what a CUDA card gives. The solver's cut
against it is Prism's own change; this is the fact it reads.
"""
from __future__ import annotations

import copy
import sys

import pytest

from tests.unit.prism._pinned_machine import APPLE_M4_PRO, V100_16GB, profile


def _metal_device_or_skip():
    if sys.platform != "darwin":
        pytest.skip("MTLDevice.maxBufferLength is asked of a Metal device")
    try:
        import Metal
        device = Metal.MTLCreateSystemDefaultDevice()
    except Exception:
        device = None
    if device is None:
        pytest.skip("no Metal device on this host")
    return device


def test_the_detection_asks_the_device():
    from neurobrix.core.prism import autodetect
    device = _metal_device_or_skip()
    assert autodetect._detect_apple_max_allocation_mb() == int(device.maxBufferLength()) // (1024 * 1024)


def test_the_detected_apple_device_carries_it():
    from neurobrix.core.prism import autodetect
    device = _metal_device_or_skip()
    apple = [d for d in autodetect._parse_system_profiler() if d.get("brand") == "apple"]
    assert apple, "system_profiler listed no Apple GPU on a host with a Metal device"
    for d in apple:
        assert d.get("max_allocation_mb") == int(device.maxBufferLength()) // (1024 * 1024), d
        assert 0 < d["max_allocation_mb"] <= d["host_memory_mb"]


def test_the_loader_carries_it_to_the_device_and_a_silent_profile_says_no_limit():
    said = copy.deepcopy(APPLE_M4_PRO)
    said["devices"][0]["max_allocation_mb"] = 13639          # this M4 Pro's maxBufferLength, 2026-10-04
    assert profile(said).devices[0].max_allocation_mb == 13639
    assert profile(APPLE_M4_PRO).devices[0].max_allocation_mb is None
    assert profile(V100_16GB).devices[0].max_allocation_mb is None
