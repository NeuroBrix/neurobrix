"""A weight set's memory (`nbxtensor.memory.weight_arenas()`): bytes counted per device and for the page-locked host
block at the arenas' alignment, one arena per device and one host block allocated once, the host block page-locked
once its weights are written.

Before (2026-11-13): the engine's Triton loader counted the aligned bytes, built the `ComponentArena`s and the
`HostPinnedArena` and pinned it itself (the red line: memory is the library's; the engine decides where each weight
goes).

What this test does if the code is wrong: a total is not aligned or lands on another device, a device with no bytes
gets an arena, the host block is made without host bytes, allocation runs twice, or the pin is skipped — red.

Run: python -m pytest nbxtensor/tests/test_a_weight_sets_memory_is_counted_then_held_by_the_library.py
"""
import pytest

from nbxtensor.memory import memory_pool as M
from nbxtensor.memory.layout import ARENA_ALIGNMENT, aligned


@pytest.fixture
def made(monkeypatch):
    seen = []

    class Device:
        ALIGNMENT = M.ComponentArena.ALIGNMENT

        def __init__(self, total, dev):
            seen.append(("device", dev, total))

    class Host:
        ALIGNMENT = M.HostPinnedArena.ALIGNMENT

        def __init__(self, total):
            seen.append(("host", total))

        def pin(self):
            seen.append(("pin",))
    monkeypatch.setattr(M, "ComponentArena", Device)
    monkeypatch.setattr(M, "HostPinnedArena", Host)
    return seen


def test_bytes_are_counted_aligned_on_their_device_and_the_host(made):
    from nbxtensor import memory
    w = memory.weight_arenas()
    w.reserve(0, 10)
    w.reserve(0, ARENA_ALIGNMENT + 1)
    w.reserve(2, 3)
    w.reserve_host(5)
    assert w.device_bytes() == {0: aligned(10, ARENA_ALIGNMENT) + aligned(ARENA_ALIGNMENT + 1, ARENA_ALIGNMENT),
                                2: aligned(3, ARENA_ALIGNMENT)}
    w.allocate()
    assert made == [("device", 0, w.device_bytes()[0]), ("device", 2, aligned(3, ARENA_ALIGNMENT)),
                    ("host", aligned(5, ARENA_ALIGNMENT))]
    assert sorted(w.devices) == [0, 2]
    w.pin_host()
    assert made[-1] == ("pin",)


def test_no_host_bytes_no_host_block_and_the_pin_is_a_no_op(made):
    from nbxtensor import memory
    w = memory.weight_arenas()
    w.reserve(1, 7)
    w.allocate()
    w.pin_host()
    assert w.host is None and made == [("device", 1, aligned(7, ARENA_ALIGNMENT))]


def test_allocation_happens_once(made):
    from nbxtensor import memory
    w = memory.weight_arenas()
    w.reserve(0, 7)
    w.allocate()
    with pytest.raises(RuntimeError, match="already allocated"):
        w.allocate()
