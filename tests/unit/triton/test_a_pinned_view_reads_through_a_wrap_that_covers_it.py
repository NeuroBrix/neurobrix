"""A pinned address is read through a wrap that COVERS it.

The pinned wraps were kept under their base address alone. A base names two different extents: a pool segment (what
the allocator's `containing()` answers for an address that is not a block's own base — a view inside a block) and the
segment's FIRST BLOCK, registered at the same address. With the first block pinned (its wrap as long as the block), a
view inside ANOTHER block of that segment resolved to the same key, found the short wrap, and `pinned_gpu_address`
answered an address past that buffer's end: wrong bytes, nothing raised. Found 2026-11-11 on the Triton expert store
(its slots are views of one arena): after nine resident mixture-of-experts runs in the process, the arena landed in a
segment whose first block — a 4 096-byte expert of an earlier run, pinned by a cached pointer table — was still
pinned, and every row of the store's result was wrong (a slot 47 616 bytes into a 4 096-byte wrap).

What these tests would do if the code were wrong: the first reads the view's bytes back wrong; the second lets an
address beyond its wrap out instead of refusing it by name.
"""
from __future__ import annotations

import numpy as np
import pytest

triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

from nbxtensor.backend.launcher import launch  # noqa: E402
from nbxtensor.dtype.nbx_dtype import NBXDtype  # noqa: E402
from nbxtensor.memory.allocator import DeviceAllocator as DA  # noqa: E402
from nbxtensor.tensor import NBXTensor  # noqa: E402

N, INSIDE = 256, 2048          # the view: N float32, INSIDE bytes into its block


def _metal_or_skip():
    try:
        from nbxtensor.backend.runtime import gpu_backend
        if gpu_backend() != "metal":
            pytest.skip("the pinned wraps are the Metal driver's")
        from nbxtensor.backend import triton_ext_driver as drv
        return drv
    except Exception:
        pytest.skip("no Metal backend the engine can resolve")


@triton.jit
def _read_through(tab_ptr, out_ptr, n, BLOCK: tl.constexpr):
    off = tl.arange(0, BLOCK)
    m = off < n
    src = tl.cast(tl.load(tab_ptr), tl.pointer_type(tl.float32), bitcast=True)
    tl.store(out_ptr + off, tl.load(src + off, mask=m, other=0.0), mask=m)


FAR = 1 << 16                  # well past the first block's own pages (16 KiB pages)


def _a_pinned_first_block_and_a_far_block_of_its_segment(drv):
    """(first, block, scopes, kept): `first` is a small block lying AT its segment's base, PINNED as soon as it is
    met — its wrap is made before the far block exists, as an earlier run's cached table does — and `block` another
    block of the same segment at least FAR bytes in. Small tensors are allocated (and kept alive) until the pool
    opens a segment and carves both."""
    kept, firsts, scopes = [], {}, []
    for i in range(4096):
        t = NBXTensor.empty((INSIDE + 4 * N,), dtype=NBXDtype.uint8, device="cuda:0")
        kept.append(t)
        base, size = DA.containing(t.data_ptr())
        if base == t.data_ptr() and DA._cuda_ptr_size[base] < size:
            firsts[base] = t
            scopes.append(drv.pinned_addresses(t))
            scopes[-1].__enter__()
        elif base in firsts and t.data_ptr() - base >= FAR:
            return firsts[base], t, scopes, kept
    for s in scopes:
        s.__exit__(None, None, None)
    pytest.skip("the pool carved no segment into a first block and a far one here")


def test_a_view_in_another_block_is_read_right_while_its_segments_first_block_is_pinned():
    import ctypes
    drv = _metal_or_skip()
    first, block, scopes, kept = _a_pinned_first_block_and_a_far_block_of_its_segment(drv)
    try:
        want = np.arange(N, dtype=np.float32) + 5000
        ctypes.memmove(block.data_ptr() + INSIDE, want.ctypes.data, 4 * N)
        view = NBXTensor.from_raw(block.data_ptr() + INSIDE, (N,), NBXDtype.float32, "cuda", owns_data=False,
                                  device_idx=0, base=block)
        assert DA.containing(view.data_ptr())[0] == first.data_ptr() != block.data_ptr()
        with drv.pinned_addresses(view):
            gpu = drv.pinned_gpu_address(view.data_ptr())
            covering = [(base, wrap.nbytes, va) for (base, _), (wrap, _, va) in drv._RESIDENT_WRAPS.items()
                        if va <= gpu and gpu + 4 * N <= va + wrap.nbytes]
            tab = NBXTensor.from_numpy(np.array([gpu], dtype=np.int64))
            out = NBXTensor.from_numpy(np.zeros(N, dtype=np.float32))
            launch(_read_through, (1,), tab, out, N, BLOCK=N)
            got = np.asarray(out.numpy()).copy()
    finally:
        for s in scopes:
            s.__exit__(None, None, None)
    # The contract, not the accident: the address lies inside a held wrap, at the view's own offset in it. (Past a
    # wrap's end the read is whatever Metal maps there — the neighbour's bytes with 240 wraps alive, the right ones
    # in a quiet process: a test on that alone is a coin.)
    assert covering, f"the pinned address {gpu:#x} lies in no held wrap: {sorted(drv._RESIDENT_WRAPS)[:4]}"
    base, _, va = covering[0]
    assert gpu - va == view.data_ptr() - base
    assert np.array_equal(got, want), f"the view read {got[:4]} through its pinned address, it holds {want[:4]}"
    del kept


def test_an_address_beyond_the_wrap_it_resolves_to_is_refused_by_name():
    """Not reachable through the allocator once the wraps are keyed by their extent; the refusal is what stands
    between a future disagreement of the two and an address past a buffer's end. The held wrap is replaced by a
    shorter one to see it."""
    drv = _metal_or_skip()
    t = NBXTensor.from_numpy(np.zeros(1024, np.float32))
    inner = NBXTensor.from_raw(t.data_ptr() + 64, (16,), NBXDtype.float32, "cuda", owns_data=False, device_idx=0,
                               base=t)

    class Short:
        nbytes = 16

    with drv.pinned_addresses(inner):
        key = drv._pinned_extent(inner.data_ptr())
        assert drv.pinned_gpu_address(inner.data_ptr()) == drv._RESIDENT_WRAPS[key][2] + inner.data_ptr() - key[0]
        real = drv._RESIDENT_WRAPS[key]
        drv._RESIDENT_WRAPS[key] = [Short(), real[1], real[2]]     # put back below: the scope's exit ends the real one
        try:
            with pytest.raises(RuntimeError, match="beyond the wrap"):
                drv.pinned_gpu_address(inner.data_ptr())
        finally:
            drv._RESIDENT_WRAPS[key] = real
    assert key not in drv._RESIDENT_WRAPS
