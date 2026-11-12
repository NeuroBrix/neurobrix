"""`Arena.bytes_on(device)`: what an arena's slots hold on a device, the weight slots apart from the rest, each
allocation counted once. The engine's memory ledger (NBX_LIVE_WATERMARK_TRACE) summed the slots itself through the
arena's private list until 2026-11-12.

What this test would do if the code were wrong: a view counted beside the tensor it views, a weight counted with the
intermediates, a tensor of another device counted, an empty slot read — RED."""
import numpy as np
import pytest

from nbxtensor.memory.allocator import DeviceAllocator as DA
from nbxtensor.memory.arena import Arena
from nbxtensor.tensor import NBXTensor


def test_the_weights_and_the_rest_are_counted_apart_once_per_allocation():
    if DA.device_count() == 0:          # only a machine without an accelerator skips; any other error fails
        pytest.skip("no accelerator on this machine")
    dev = DA.get_device()
    w = NBXTensor.from_numpy(np.zeros((4, 8), np.float32))          # 128 bytes
    x = NBXTensor.from_numpy(np.zeros((2, 8), np.float16))          # 32 bytes
    arena = Arena(6, num_weights=2, num_inputs=1)
    arena[0], arena[1] = w, w.t()             # a weight and a view of it: one allocation
    arena[2], arena[3], arena[4] = x, None, x  # an input, an empty slot, the input again as an intermediate
    assert arena.bytes_on(dev) == (128, 32)
    assert arena.bytes_on(dev + 1) == (0, 0)
