"""A pinned address is read through the buffer that COVERS it.

(Until 2026-11-12 the address came from a wrap of the memory; nbxtensor binds its own buffers now, and the address is
the allocation's buffer address plus the offset. The defect below is what the test keeps shut.)

The pinned wraps were kept under their base address alone. A base names two different extents: a pool segment (what
the allocator's `containing()` answers for an address that is not a block's own base — a view inside a block) and the
segment's FIRST BLOCK, registered at the same address. With the first block pinned (its wrap as long as the block), a
view inside ANOTHER block of that segment resolved to the same key, found the short wrap, and `pinned_gpu_address`
answered an address past that buffer's end: wrong bytes, nothing raised. Found 2026-11-11 on the Triton expert store
(its slots are views of one arena): after nine resident mixture-of-experts runs in the process, the arena landed in a
segment whose first block — a 4 096-byte expert of an earlier run, pinned by a cached pointer table — was still
pinned, and every row of the store's result was wrong (a slot 47 616 bytes into a 4 096-byte wrap).

What these tests would do if the code were wrong: the first reads the view's bytes back wrong; the second lets an
address beyond its allocation out instead of refusing it by name.
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
            from nbxtensor.backend.metal_device import runtime
            buffer, offset = runtime().buffer_for_pointer(view.data_ptr())
            tab = NBXTensor.from_numpy(np.array([gpu], dtype=np.int64))
            out = NBXTensor.from_numpy(np.zeros(N, dtype=np.float32))
            launch(_read_through, (1,), tab, out, N, BLOCK=N)
            got = np.asarray(out.numpy()).copy()
    finally:
        for s in scopes:
            s.__exit__(None, None, None)
    # The contract, not the accident: the address is the view's own allocation's buffer address at the view's offset
    # in it, inside that buffer. (Past a buffer's end the read is whatever Metal maps there: a test on the bytes
    # alone is a coin.)
    assert buffer is not None and gpu == int(buffer.gpuAddress()) + offset, (hex(gpu), offset)
    assert offset + 4 * N <= int(buffer.length())
    assert np.array_equal(got, want), f"the view read {got[:4]} through its pinned address, it holds {want[:4]}"
    del kept


def test_an_address_beyond_its_allocation_is_refused_by_name():
    """An address past the end of every allocation has no shader address: refused by name, never answered with an
    address past a buffer's end."""
    drv = _metal_or_skip()
    from nbxtensor.backend.metal_device import runtime
    t = NBXTensor.from_numpy(np.zeros(1024, np.float32))
    buffer, offset = runtime().buffer_for_pointer(t.data_ptr())
    past = t.data_ptr() - offset + int(buffer.length())               # the first byte after the allocation's buffer
    if runtime().buffer_for_pointer(past)[0] is not None:
        pytest.skip("another allocation begins right after this one")
    with pytest.raises(RuntimeError, match="in no allocation of this Metal runtime"):
        runtime().gpu_address(past)
