"""`memory_pool.release_weights`, `memory_pool.join_weights` and `Arena.release`: a loaded weight set (NBXTensors, its
arenas under "_arenas") released after every device holding it synchronised, the library's caches that hold weights
dropped and the pool emptied; a second load's arenas joined to the first's so no weight points into a freed block;
an arena's slots dropped after their devices drain. The NeuroBrix engine did each of these itself until 2026-11-12.

What this test would do if the code were wrong: a device left unsynchronised before the references drop, another
library's weights cleared, the pool not emptied, a second load's arenas replacing the first's (or an empty one doing
so), an arena cleared before its devices synchronise — RED."""
import gc

import numpy as np
import pytest

from nbxtensor.memory import deferred_frees, memory_pool
from nbxtensor.memory.allocator import DeviceAllocator as DA
from nbxtensor.memory.arena import Arena
from nbxtensor.tensor import NBXTensor


class _Arena:
    freed = []

    def __init__(self, device_idx=None):
        if device_idx is not None:
            self.device_idx = device_idx

    def __del__(self):
        _Arena.freed.append(id(self))


@pytest.fixture
def driver(monkeypatch):
    calls = []
    monkeypatch.setattr(DA, "device_synchronize", staticmethod(lambda d=None: calls.append(("sync", d))))
    monkeypatch.setattr(DA, "empty_cache_pool", staticmethod(lambda: calls.append(("empty",)) or 0))
    return calls


def test_the_library_s_weights_are_released_after_every_device_holding_them_drains(driver):
    if DA.device_count() == 0:          # only a machine without an accelerator skips; any other error fails
        pytest.skip("no accelerator on this machine")

    class Freed:
        def __init__(self, device_idx):
            self.device_idx = device_idx

        def __del__(self):
            driver.append(("freed", self.device_idx))

    w = NBXTensor.from_numpy(np.zeros(4, np.float32))
    weights = {"w": w, "_arenas": {"3": Freed(3), "3#1": _Arena()}}
    assert memory_pool.release_weights(weights) is True and weights == {}
    assert driver == [("sync", d) for d in sorted({3, w.device_idx})] + [("freed", 3), ("empty",)], driver


def test_another_library_s_weights_are_left_alone_and_an_empty_arena_set_drains_nothing(driver):
    torch_like = {"w": object()}
    assert memory_pool.release_weights(torch_like) is False and "w" in torch_like and driver == []
    empty = {"_arenas": {}}
    assert memory_pool.release_weights(empty) is True and empty == {} and driver == []


def test_a_second_load_joins_its_arenas_instead_of_replacing_the_first():
    _Arena.freed.clear()
    first, second = _Arena(), _Arena()
    held = {"w0": "t0", "_arenas": {2: first}}
    held.update(memory_pool.join_weights(held, {"w1": "t1", "_arenas": {2: second}}))
    first_id = id(first)
    del first, second
    gc.collect()
    assert first_id not in _Arena.freed and set(held) == {"w0", "w1", "_arenas"}
    assert sorted(map(str, held["_arenas"])) == ["2", "2#1"]
    assert memory_pool.join_weights(held, {"w2": "t2"}) == {"w2": "t2"}
    held.update(memory_pool.join_weights(held, {"w3": "t3", "_arenas": {}}))
    assert len(held["_arenas"]) == 2, "an empty arena set replaced the first load's"


def test_an_arena_is_cleared_after_its_devices_drain(monkeypatch):
    order = []
    monkeypatch.setattr(deferred_frees, "sync_devices_of", lambda ts: order.append(("sync", len(list(ts)))))
    arena = Arena(3, num_weights=1)
    arena[0], arena[2] = "w", "x"
    clear_all = arena.clear_all
    monkeypatch.setattr(Arena, "clear_all", lambda self: (order.append(("clear",)), clear_all()))
    epoch = arena.epoch
    arena.release()
    assert order == [("sync", 2), ("clear",)] and arena[0] is None and arena[2] is None and arena.epoch == epoch + 1


def test_the_application_s_door_answers_the_same(driver):
    import nbxtensor.memory as memory
    other = {"w": object()}
    assert memory.release_weights(other) is False and "w" in other
    first = _Arena()
    held = {"_arenas": {0: first}}
    assert memory.join_weights(held, {"_arenas": {0: _Arena()}})["_arenas"][0] is first
