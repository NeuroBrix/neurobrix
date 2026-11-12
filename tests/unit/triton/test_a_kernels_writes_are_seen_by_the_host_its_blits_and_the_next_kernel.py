"""What the host, our blits and the next kernel see of a kernel's writes on Metal.

Three crossings, one test each:
  kernel -> host    a kernel writes, the host reads the same memory
  kernel -> blit    a kernel writes, our blit copies it, the host reads the copy
  blit -> kernel    our blit writes, and a kernel reads what it wrote

Until 2026-11-12 triton-ext ran the kernels on a queue of its own, and these crossings were ordered by a drain of our
queue before every launch and a whole-device wait after it (82 % of a decode's wall time). Now nbxtensor encodes
kernels and blits on ONE serial stream (`nbxtensor/backend/metal_dispatch.py`): Metal orders them, and the host waits
only where it reads. This file says whether the three still hold; each was seen red when its ordering was removed
(nbxtensor/tests/test_a_metal_kernel_runs_on_nbxtensors_own_stream.py removes them one by one).
"""
from __future__ import annotations

import numpy as np
import pytest

triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

from nbxtensor.tensor import NBXTensor  # noqa: E402
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.backend.launcher import launch  # noqa: E402


@triton.jit
def _stamp(out_ptr, value, N, BLOCK: tl.constexpr):
    """Write `value` into every element — cheap, and big enough to still be in
    flight when the host looks if nothing ordered it."""
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    tl.store(out_ptr + o, tl.full((BLOCK,), 1, tl.int32) * value, mask=m)


def _backend_or_skip():
    try:
        from nbxtensor.backend.runtime import gpu_backend
        if gpu_backend() != "metal":
            pytest.skip("these crossings are checked on Metal")
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


@triton.jit
def _sum_into(src_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """Read what our blit wrote and record it, so a stale read is visible."""
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    v = tl.load(src_ptr + o, mask=m, other=0)
    tl.store(out_ptr + o, v, mask=m)


def test_their_kernel_sees_what_our_blit_just_wrote():
    """The third crossing: our queue writes, their queue reads.

    This is the direction `_nbx_queue_drain()` guards, and the only one of the
    three that the per-launch `_native().synchronize()` does NOT cover — that
    wait orders their queue against us, not us against their queue.
    """
    _backend_or_skip()
    for rep in range(REPS):
        value = 0x11110000 | rep
        # a source our blit will copy FROM, filled by the host
        src = NBXTensor.from_numpy(np.full(N, value, dtype=np.int32))
        # the buffer their kernel will read: written ONLY by our blit
        mid = NBXTensor.from_numpy(np.zeros(N, dtype=np.int32))
        out = NBXTensor.from_numpy(np.zeros(N, dtype=np.int32))
        DeviceAllocator.memcpy(mid.data_ptr(), src.data_ptr(), N * 4, 3)  # 3 = D2D
        # No wait here on purpose: the drain inside the launch path is what
        # must make the blit visible to the kernel that reads `mid`.
        launch(_sum_into, (triton.cdiv(N, BLOCK),), mid, out, N, BLOCK=BLOCK)
        got = out.numpy()
        assert int((got != value).sum()) == 0, (
            f"rep {rep}: their kernel read {int((got != value).sum())} of {N} "
            f"elements our blit had already written — the queues are not ordered")
