"""`nbxtensor.memory.memory_pool.break_aliases`: before tensors are freed, every other entry of the same list or dict
that views one of their allocations is replaced by its `contiguous()`; the engine's weight eviction (zero3) calls it
over its arena or its store. It lived in the engine twice (triton/sequence.py and triton/sequential.py) until
2026-11-12.

What this test would do if the code were wrong: a strided view of a freed weight left in place (it would read freed
memory), a view of a kept tensor copied for nothing, a freed entry itself replaced, a None or a non-tensor entry
touched — RED."""
import numpy as np
import pytest

from nbxtensor.memory.memory_pool import break_aliases
from nbxtensor.tensor import NBXTensor


def _accelerator():
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() == 0:          # only a machine without an accelerator skips; any other error fails
        pytest.skip("no accelerator on this machine")


def _weights():
    return (NBXTensor.from_numpy(np.arange(12, dtype=np.float32).reshape(3, 4)),
            NBXTensor.from_numpy(np.ones((2, 2), np.float32)))


def test_a_list_entry_viewing_a_freed_weight_is_copied_and_the_rest_is_left():
    _accelerator()
    w, kept = _weights()
    view_of_w, view_of_kept = w.t(), kept.t()
    arena = [w, kept, view_of_w, view_of_kept, None, 7]
    assert break_aliases(arena, [0]) == 1
    assert arena[0] is w and arena[1] is kept and arena[3] is view_of_kept and arena[4] is None and arena[5] == 7
    assert arena[2] is not view_of_w and arena[2].storage_root() is arena[2]          # its own allocation now
    assert np.array_equal(arena[2].numpy(), np.arange(12, dtype=np.float32).reshape(3, 4).T)


def test_a_dict_store_is_answered_the_same_and_nothing_freed_copies_nothing():
    _accelerator()
    w, _ = _weights()
    store = {"param::w": w, "act": w.t(), "gone": None}
    assert break_aliases(store, ["param::missing"]) == 0
    assert break_aliases(store, ["param::w"]) == 1
    assert store["act"].storage_root() is store["act"] and store["param::w"] is w
