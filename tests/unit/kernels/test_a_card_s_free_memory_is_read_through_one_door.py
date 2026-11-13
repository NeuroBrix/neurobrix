"""`DeviceAllocator.free_memory_mb(index)` answers the driver's free memory for
ONE card, in MB, without torch — the planner's door to the machine as it is.

Seen needed 2026-09-20: the plan's `free_mb` was its own accounting, so a
neighbour holding 18-27 GB of a V100-32GB was invisible until the first
allocation failed (five of five runs). Skips where no GPU runtime answers.
"""
from __future__ import annotations

import pytest

from nbxtensor.memory.allocator import DeviceAllocator as D


def test_free_is_positive_and_at_most_the_card():
    try:
        n = D.device_count()
    except Exception:
        n = 0
    if not n:
        pytest.skip("no GPU runtime here")
    free = D.free_memory_mb(0)
    assert free is not None and 0 < free
    # a V100 carries 16 or 32 GB; any card answers less than its own total
    assert free <= 256 * 1024
    # the door restores the previously-current device
    assert D.get_device() == D.get_device()


def test_a_device_the_process_does_not_see_is_refused_by_name():
    """Until 2026-11-13 it answered None, and Prism read None as a card of 0 MB (qa)."""
    from nbxtensor.memory.allocator import DeviceQueryError
    with pytest.raises(DeviceQueryError, match=r"device 1000000: the process sees \d+ device"):
        D.free_memory_mb(10**6)
