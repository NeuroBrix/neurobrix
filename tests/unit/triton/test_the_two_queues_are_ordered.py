"""Two queues, one device: what the host and our blits see of a kernel's writes.

triton-ext dispatches kernels on ITS OWN MTLCommandQueue. NeuroBrix's allocator
has another, on which it enqueues device-to-device blits. Metal orders command
buffers WITHIN a queue (commit order) and orders NOTHING across two of them
without an explicit event or a host wait.

The driver pays for that ordering today with a host wait after EVERY launch
(`_native().synchronize()`), which is measured at 82% of a decode's wall time.
Anything that replaces it must carry the same guarantee, and this file is what
says whether it does.

TWO CROSSINGS, one test each:
  ext -> host    a kernel writes, the host reads the same memory
  ext -> blit    a kernel writes, our queue copies it, the host reads the copy

Both are written so they FAIL when the ordering is removed. That is checked in
`test_the_guard_is_what_makes_these_pass`, which strips the driver's wait and
asserts the values go wrong — a test that cannot fail proves nothing, and this
one was verified to fail before it was trusted.
"""
from __future__ import annotations

import numpy as np
import pytest

triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator  # noqa: E402
from neurobrix.kernels.launcher import launch  # noqa: E402


@triton.jit
def _stamp(out_ptr, value, N, BLOCK: tl.constexpr):
    """Write `value` into every element — cheap, and big enough to still be in
    flight when the host looks if nothing ordered it."""
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    tl.store(out_ptr + o, tl.full((BLOCK,), 1, tl.int32) * value, mask=m)


def _backend_or_skip():
    try:
        from neurobrix.kernels.nbx_tensor import _detect_gpu_backend
        if _detect_gpu_backend() != "metal":
            pytest.skip("the two-queue hazard is Metal's; there is one queue elsewhere")
    except Exception:
        pytest.skip("no GPU backend the engine can resolve")


N = 1 << 20          # 4 MB of int32: large enough that a lost ordering shows
BLOCK = 1024
REPS = 8             # one shot can pass by luck; a lost ordering shows in a run


def _stamped(value: int) -> NBXTensor:
    t = NBXTensor.from_numpy(np.zeros(N, dtype=np.int32))
    launch(_stamp, (triton.cdiv(N, BLOCK),), t, value, N, BLOCK=BLOCK)
    return t


def test_the_host_sees_what_the_kernel_just_wrote():
    """ext -> host. The host reads the very memory the kernel wrote."""
    _backend_or_skip()
    for rep in range(REPS):
        value = 0x5A5A0000 | rep      # 0x5A5A0000 = 1515847680, inside int32
        t = _stamped(value)
        got = t.numpy()
        bad = int((got != value).sum())
        assert bad == 0, (
            f"rep {rep}: {bad}/{N} elements were not the value the kernel "
            f"stored — the host read memory the GPU had not finished writing")


def test_our_blit_sees_what_the_kernel_just_wrote():
    """ext -> our queue. A device-to-device copy on the ALLOCATOR's queue reads
    a buffer the ext queue wrote. Nothing orders those two queues by itself."""
    _backend_or_skip()
    for rep in range(REPS):
        # fits in int32: 0xA5A50000 is 2779086848 and wraps negative,
        # which made this assertion unfalsifiable rather than strict
        value = 0x2A5A0000 | rep
        src = _stamped(value)
        dst = NBXTensor.from_numpy(np.zeros(N, dtype=np.int32))
        DeviceAllocator.memcpy(dst.data_ptr(), src.data_ptr(), N * 4, 3)   # 3 = D2D
        got = dst.numpy()
        bad = int((got != value).sum())
        assert bad == 0, (
            f"rep {rep}: {bad}/{N} elements of the COPY were not the value the "
            f"kernel stored — our queue read what the ext queue had not finished")
