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
from neurobrix.triton.weight_loader import _split_across_workers


def test_the_parallel_widening_is_the_old_formula_bit_for_bit():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2 ** 16, size=3_000_001, dtype=np.uint64).astype(np.uint16)   # every class: nan, inf, subnormal
    bits[:4] = [0x7F80, 0xFF80, 0x7FC0, 0x0001]
    old = (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float16)
    new = _split_across_workers(bits, np.empty(bits.shape, dtype=np.float16), bf16_bits_to_f16)
    assert new.dtype == np.float16
    assert np.array_equal(old.view(np.uint16), new.reshape(-1).view(np.uint16))


def _gpu():
    try:
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _gpu(), reason="page-locked memory needs the device runtime")
def test_host_weights_share_one_pinned_block_that_outlives_the_load():
    from neurobrix.kernels.nbx_tensor import DeviceAllocator, NBXDtype
    from neurobrix.triton.memory_pool import HostPinnedArena
    from neurobrix.triton.weight_loader import _load_to_pinned_cpu
    rng = np.random.default_rng(1)
    raws = [rng.integers(0, 2 ** 16, size=(64, 33), dtype=np.uint64).astype(np.uint16) for _ in range(5)]
    before = len(DeviceAllocator._host_pinned_ptr_size)
    arena = HostPinnedArena(sum(-(-r.size * 2 // 256) * 256 for r in raws))
    ts = [_load_to_pinned_cpu(r.tobytes(), r.shape, NBXDtype.bfloat16, NBXDtype.float16, arena) for r in raws]
    assert len(DeviceAllocator._host_pinned_ptr_size) == before + 1, "a pinned block per weight"
    del arena
    for r, t in zip(raws, ts):
        want = (r.astype(np.uint32) << 16).view(np.float32).astype(np.float16)
        assert np.array_equal(t.numpy().view(np.uint16), want.view(np.uint16))
    del ts, t                       # the loop above left the last weight bound to `t`
    assert len(DeviceAllocator._host_pinned_ptr_size) == before, "the block outlived its last weight"
