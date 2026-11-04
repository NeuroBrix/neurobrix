"""A replay recording is sized against what the device can give it (`replay.recording_room`): its free memory, and on a
device that draws from the host's memory, no more than the machine has available.

The Mac's RUN 29 (2026-11-04 19:56): Metal answers `mem_get_info` with the recommended working set less what the process
allocated — a budget's headroom, not memory the machine has. Allegro-TI2V's 6.6 GB encoder slab fitted that headroom,
was frozen, and its replay died at its 23rd launch, the command buffer refused for insufficient memory.

What this test would do if the code were wrong: the device's headroom alone on a unified device — red at the first
test; a discrete card bounded by the host's RAM (a 32 GB V100 in a machine with 16 GB free) — red at the second; a
unified machine whose free memory cannot be read sized by a guess — red at the third. Card-less."""
from __future__ import annotations

import pytest

from neurobrix.triton import replay as R

GB = 1 << 30


def _machine(monkeypatch, device_free, unified, available_mb):
    from nbxtensor.backend import device_facts, host_memory
    monkeypatch.setattr(R.DeviceAllocator, "mem_get_info", staticmethod(lambda dev=None: (device_free, 2 * device_free)))
    monkeypatch.setattr(device_facts, "host_shares_memory_with_device", lambda: unified)
    monkeypatch.setattr(host_memory, "memory_state",
                        lambda: host_memory.MemoryState(available_mb=available_mb, source="pinned for the test"))


def test_a_unified_device_records_within_the_machines_free_memory(monkeypatch):
    _machine(monkeypatch, 18 * GB, True, 10 * 1024)
    assert R.recording_room(0) == 10 * GB


def test_a_discrete_card_records_within_its_own_free_memory(monkeypatch):
    _machine(monkeypatch, 30 * GB, False, 16 * 1024)
    assert R.recording_room(0) == 30 * GB


def test_a_unified_machine_that_cannot_be_read_is_refused(monkeypatch):
    _machine(monkeypatch, 18 * GB, True, None)
    with pytest.raises(RuntimeError, match="ZERO FALLBACK: device 0 shares the host's memory"):
        R.recording_room(0)
