"""`oom_advice.held_memory_report` and `Arena.census`: after a refused allocation, the lines that say who holds the
device's memory — the blocks in use by size, the tensors alive on the large ones (named by the application, or ORPHAN)
with what refers to them, the autotuners still holding tensors — and what an arena holds by region. The engine's
NBX_LIVE_DUMP_ON_OOM dump walked the allocator's private tables and the garbage collector itself (330 lines) until
2026-11-12.

What this test would do if the code were wrong: a large block left out of its device's line, a named tensor called an
orphan or an orphan given a name, its holder not said, a small tensor listed as large, a slot counted in the wrong
region, a large intermediate missed or a weight listed as one — RED."""
import numpy as np
import pytest

from nbxtensor.memory.allocator import DeviceAllocator as DA
from nbxtensor.memory.arena import Arena
from nbxtensor.memory.oom_advice import held_memory_report
from nbxtensor.tensor import NBXTensor

MB = 1 << 20


def _accelerator():
    if DA.device_count() == 0:          # only a machine without an accelerator skips; any other error fails
        pytest.skip("no accelerator on this machine")


def test_the_large_tensors_are_named_or_orphaned_with_their_holders():
    _accelerator()
    named = NBXTensor.from_numpy(np.zeros(5 * MB // 4, np.float32))         # 5 MB
    keeper = {"the_orphan_keeper": NBXTensor.from_numpy(np.zeros(6 * MB // 4, np.float32))}
    small = NBXTensor.from_numpy(np.zeros(16, np.float32))
    lines = held_memory_report([("act::named", named), ("act::small", small)], big=4 * MB, bucket=4 * MB)
    text = "\n".join(lines)
    dev = DA.get_device()
    assert any(l.startswith(f"[LIVE_BLOCKS dev={dev}]") and "5MB" in l and "6MB" in l for l in lines), text
    mine = [l for l in lines if "tensor#" in l and ("tid=act::named" in l or "ORPHAN" in l and "nbytes=6MB" in l)]
    assert len(mine) == 2, text
    orphan = next(l for l in mine if "ORPHAN" in l)
    assert "dict[len=1]" in orphan, orphan
    assert "act::small" not in text
    assert any(l.startswith("[GC_COLLECT_TEST]") for l in lines) and any(l.startswith("[BIG_TENSORS]") for l in lines)
    del keeper


def test_an_arena_is_counted_by_region_and_its_large_intermediates_listed():
    _accelerator()
    w = NBXTensor.from_numpy(np.zeros(3 * MB // 4, np.float32))
    x = NBXTensor.from_numpy(np.zeros(4, np.float32))
    big = NBXTensor.from_numpy(np.zeros(2 * MB // 4, np.float32))
    arena = Arena(5, num_weights=1, num_inputs=1)
    arena[0], arena[1], arena[2], arena[3] = w, x, big, x
    c = arena.census(MB)
    assert c["weights"] == [1, 3 * MB] and c["inputs"] == [1, 16] and c["intermediates"] == [2, 2 * MB + 16]
    assert c["empty"] == 1 and c["other"] == 0 and c["types"] == {"NBXTensor": 4}
    assert c["large"] == [(2, big)]
