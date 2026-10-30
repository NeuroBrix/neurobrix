"""The Triton loader's host-held weights (zero3's block weights, an expert store's experts): decoded across the
loader's workers and placed in ONE page-locked block (`HostPinnedArena`). Qwen3-30B-A3B triton loaded 57 GB to the
host in 447 s on the V100 rack (2026-10-27): py-spy put ~40 % in 18 864 per-weight cudaMallocHost calls and ~40 %
in a one-thread bf16 widening.

What each test would do if the code were wrong: a widening that rounds otherwise than the old formula differs on
some bit pattern (red); a host weight not taken from the one block — a second block per weight (red); a host
weight that does not keep its block alive reads freed memory once the load's locals are gone (red)."""
from __future__ import annotations

import numpy as np
import pytest

from nbxtensor.dtype.convert import bf16_bits_to_f16
from neurobrix.triton import weight_loader as W
from nbxtensor.io.safetensors import split_across_workers


def _stage(raw, shape, source, target, host_arena=None):
    """A host-held weight staged as the engine's loader stages it: nbxtensor's primitive, the loader's I/O pool."""
    from neurobrix.triton import weight_loader as W
    from nbxtensor.io import safetensors as SF
    return SF.load_to_pinned_host(raw, shape, source, target, W._io_pool(), host_arena)



def test_the_parallel_widening_is_the_old_formula_bit_for_bit():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2 ** 16, size=3_000_001, dtype=np.uint64).astype(np.uint16)   # every class: nan, inf, subnormal
    bits[:4] = [0x7F80, 0xFF80, 0x7FC0, 0x0001]
    old = (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float16)
    new = split_across_workers(bits, np.empty(bits.shape, dtype=np.float16), bf16_bits_to_f16, W._io_pool())
    assert new.dtype == np.float16
    assert np.array_equal(old.view(np.uint16), new.reshape(-1).view(np.uint16))


def _gpu():
    try:
        from nbxtensor.memory.allocator import DeviceAllocator
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _gpu(), reason="page-locked memory needs the device runtime")
def test_host_weights_share_one_pinned_block_that_outlives_the_load():
    from nbxtensor.memory.allocator import DeviceAllocator
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.memory.memory_pool import HostPinnedArena
    rng = np.random.default_rng(1)
    raws = [rng.integers(0, 2 ** 16, size=(64, 33), dtype=np.uint64).astype(np.uint16) for _ in range(5)]
    before = len(DeviceAllocator._host_pinned_ptr_size)
    arena = HostPinnedArena(sum(-(-r.size * 2 // 256) * 256 for r in raws))
    ts = [_stage(r.tobytes(), r.shape, NBXDtype.bfloat16, NBXDtype.float16, arena) for r in raws]
    arena.pin()                     # the loader's order: every weight written, then the block page-locked once
    assert len(DeviceAllocator._host_pinned_ptr_size) == before + 1, "a pinned block per weight"
    del arena
    for r, t in zip(raws, ts):
        want = (r.astype(np.uint32) << 16).view(np.float32).astype(np.float16)
        assert np.array_equal(t.numpy().view(np.uint16), want.view(np.uint16))
    del ts, t                       # the loop above left the last weight bound to `t`
    assert len(DeviceAllocator._host_pinned_ptr_size) == before, "the block outlived its last weight"


@pytest.mark.skipif(not _gpu(), reason="page-locked memory needs the device runtime")
def test_the_block_is_written_as_ordinary_memory_and_page_locked_once():
    """A page-locked allocation ran at 1.2 GiB/s on one thread (16 GiB in 13.0 s), writing ordinary memory then
    registering it at 4.9 s (V100 rack, 2026-10-30): on a backend that registers, nothing is page-locked before
    `pin()`, and `pin()` page-locks the whole block once — a second call does nothing."""
    from nbxtensor.backend import runtime as _rt
    from nbxtensor.memory.allocator import DeviceAllocator
    from nbxtensor.memory.memory_pool import HostPinnedArena
    if not _rt.device_memory().supports("host_register"):
        pytest.skip("this backend allocates the block page-locked")
    before = dict(DeviceAllocator._host_pinned_ptr_size)
    arena = HostPinnedArena(3 << 20)
    assert DeviceAllocator._host_pinned_ptr_size == before, "page-locked before the load wrote it"
    p = arena.alloc(1 << 20)
    assert p % HostPinnedArena.ALIGNMENT == 0
    arena.pin()
    arena.pin()
    added = set(DeviceAllocator._host_pinned_ptr_size) - set(before)
    assert len(added) == 1 and DeviceAllocator._host_pinned_ptr_size[added.pop()] == 3 << 20
    assert DeviceAllocator.holds(p), "a kernel may read the block: holds() must know it"
    del arena
    assert DeviceAllocator._host_pinned_ptr_size == before, "the registration outlived its block"

