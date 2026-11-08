"""An Apple profile never loses its device's largest single allocation silently.

`max_allocation_mb` (Metal's `MTLDevice.maxBufferLength`, 13 639 MB on an M4 Pro whose working set is 18 186 MB) is what
Prism's arena bound reads (`solver._max_allocation_mb`); an absent key means "no limit below memory_mb", which is a CUDA
card's truth and an Apple device's lie. The Mac, 2026-11-08: its generated `default-9f169c79` (regenerated 2026-11-04)
carried no key, Prism let a 13 912.6 MiB arena through and Open-Sora-v2's Triton cell failed at weight loading
("GPU malloc failed (error 1) for 14588372992 bytes [... driver_free=18138MB ...]"), green on 2026-10-14. The detection had
swallowed its own failure (`except Exception: pass; return None`).

Two doors: the detection that cannot ask the device refuses by name (it never writes a profile without the fact), and
the loader refuses a unified Apple device whose profile does not carry it, naming the file and how to regenerate it.
A CUDA profile is untouched: it declares no such fact and none is asked of it.
"""
from __future__ import annotations

import copy
import sys

import pytest

from tests.unit.prism._pinned_machine import APPLE_M4_PRO, V100_16GB, profile


def test_the_detection_that_cannot_ask_the_device_refuses_by_name(monkeypatch):
    from neurobrix.core.prism import autodetect
    monkeypatch.setitem(sys.modules, "Metal", None)          # `import Metal` raises ImportError
    with pytest.raises(RuntimeError, match="maxBufferLength"):
        autodetect._detect_apple_max_allocation_mb()


def test_a_unified_apple_profile_without_the_fact_is_refused_with_the_way_out():
    silent = copy.deepcopy(APPLE_M4_PRO)
    silent["devices"][0].pop("max_allocation_mb", None)
    with pytest.raises(ValueError) as refused:
        profile(silent)
    said = str(refused.value)
    assert "max_allocation_mb" in said and "maxBufferLength" in said
    assert "regenerat" in said                              # how to get it back


def test_an_apple_profile_that_carries_it_loads_and_a_cuda_profile_is_untouched():
    assert profile(APPLE_M4_PRO).devices[0].max_allocation_mb == 13639
    assert profile(V100_16GB).devices[0].max_allocation_mb is None
