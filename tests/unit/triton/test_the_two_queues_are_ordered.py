"""Two queues, one device: what the host and our blits see of a kernel's writes.

triton-ext dispatches kernels on ITS OWN MTLCommandQueue. NeuroBrix's allocator
has another, on which it enqueues device-to-device blits. Metal orders command
buffers WITHIN a queue (commit order) and orders NOTHING across two of them
without an explicit event or a host wait.

The driver pays for that ordering today with a host wait after EVERY launch
(`_native().synchronize()`), which is measured at 82% of a decode's wall time.
Anything that replaces it must carry the same guarantee, and this file is what
says whether it does.

THREE CROSSINGS, one test each:
  ext -> host    a kernel writes, the host reads the same memory
  ext -> blit    a kernel writes, our queue copies it, the host reads the copy
  blit -> ext    OUR queue's blit writes, and THEIR kernel reads what it wrote

The third was missing until 2026-09-18 and is the one the pre-launch
`_nbx_queue_drain()` exists for. Its absence was not harmless: with only the
first two, stripping the drain and keeping the synchronize left the file GREEN,
which would have licensed removing a guard that is load-bearing. D2D copies are
blits on our queue — 1086 of them in an 8-token decode — so this crossing is
taken on every real run.

All three are written so they FAIL when the ordering is removed, and that is
checked HERE rather than asserted: `test_the_drain_is_what_makes_the_third_pass`
removes the drain and requires the third crossing to go wrong. Until 2026-09-18
this docstring named a test of that kind which did not exist in the file — the
claim was made and never kept.
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


def test_the_drain_is_what_makes_the_third_pass(monkeypatch):
    """A test that cannot fail proves nothing — so remove the guard and require
    the failure.

    `_nbx_queue_drain` is replaced with a no-op for the duration. If the third
    crossing still comes out clean, then either Metal has begun ordering the two
    queues on this machine or the blit is completing too fast to catch, and in
    EITHER case the test above has stopped proving what it claims. That is worth
    a failure here, because the guard it licenses costs 2.25 s of a 17.5 s
    decode and should not be kept on an assumption.
    """
    _backend_or_skip()
    from neurobrix.triton import triton_ext_driver as drv
    monkeypatch.setattr(drv, "_nbx_queue_drain", lambda: None)
    corrupted = 0
    for rep in range(REPS):
        value = 0x22220000 | rep
        src = NBXTensor.from_numpy(np.full(N, value, dtype=np.int32))
        mid = NBXTensor.from_numpy(np.zeros(N, dtype=np.int32))
        out = NBXTensor.from_numpy(np.zeros(N, dtype=np.int32))
        DeviceAllocator.memcpy(mid.data_ptr(), src.data_ptr(), N * 4, 3)
        launch(_sum_into, (triton.cdiv(N, BLOCK),), mid, out, N, BLOCK=BLOCK)
        if int((out.numpy() != value).sum()):
            corrupted += 1
    assert corrupted, (
        "with the drain removed, all %d reps still read what our blit wrote: "
        "the third crossing is no longer falsifiable and the guard it licenses "
        "needs re-justifying" % REPS)
