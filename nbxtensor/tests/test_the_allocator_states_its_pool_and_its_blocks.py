"""`DeviceAllocator.pool_cached_bytes`, `pool_enabled`, `pool_stats`, `blocks_in_use`: the allocator's figures an
application reads (a memory line, a forensic dump, a certifier's per-key state) without reaching its private tables.
Until 2026-11-12 the engine read `_cuda_live_bytes`, `_POOL.cached`, `_POOL.free`, `_cuda_ptr_size` and `_pool_enabled`
itself, in a dozen places.

What this test would do if the code were wrong: a block a tensor holds missing from `blocks_in_use` or listed at a
size other than its own, a freed block still listed as in use, the parked bytes not following the free, the counters
handed out as the live dict a caller could corrupt, the switch read before the allocator read it — RED."""
import pytest

from nbxtensor.memory.allocator import DeviceAllocator as DA


def _accelerator():
    if DA.device_count() == 0:          # only a machine without an accelerator skips; any other error fails
        pytest.skip("no accelerator on this machine")


def test_a_held_block_is_listed_at_its_size_and_a_freed_one_is_parked_not_listed():
    _accelerator()
    dev = DA.get_device()
    nbytes = 3 << 20
    ptr = DA.malloc_cuda(nbytes)
    try:
        held = DA.blocks_in_use(dev)
        assert held.get(ptr, 0) >= nbytes and ptr in DA.blocks_in_use()
        assert ptr not in DA.blocks_in_use(dev + 1)
        parked_before = DA.pool_cached_bytes(dev)
    finally:
        DA.free_cuda(ptr)
    assert ptr not in DA.blocks_in_use(dev)
    if DA.pool_enabled():
        assert DA.pool_cached_bytes(dev) >= parked_before + nbytes
        assert DA.pool_cached_bytes() >= DA.pool_cached_bytes(dev)
    else:
        assert DA.pool_cached_bytes(dev) == parked_before


def test_the_counters_are_a_copy_and_the_switch_is_the_allocator_s_own():
    _accelerator()
    st = DA.pool_stats()
    assert {"exact", "fit", "miss", "flushes", "evictions"} <= set(st)
    st["flushes"] = -1
    assert DA.pool_stats()["flushes"] != -1
    assert DA.pool_enabled() is DA._pool_enabled and isinstance(DA.pool_enabled(), bool)
