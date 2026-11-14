"""`NBXTensor.from_numpy` (through `DeviceAllocator.upload`, `memory.staging`): a small host array copied to the card
while the card is busy no longer holds the host until the card's queue drains. Measured 2026-11-13 on a V100
(Qwen2.5-Coder-7B fp16): ~17 synchronous uploads a decode token, 165 us of host time each, the host idle while the
card emptied its queue. The bytes go through a page-locked ring and an asynchronous copy on the stream every kernel
is launched on.
What this would do if the code were wrong: a ring region rewritten before its copy ran — a position, a length or a
table of an EARLIER call read by a later kernel: a wrong token with no error; or a caller's array read after the call
returned — the value it was changed to afterwards."""
import ctypes
import time

import numpy as np
import pytest

from nbxtensor.backend import runtime as _rt
from nbxtensor.memory import staging
from nbxtensor.memory.allocator import AllocatorInterceptor, DeviceAllocator
from nbxtensor.tensor import NBXTensor

S = staging.STAGING


def _busy(seconds_hint_copies: int = 40, nbytes: int = 256 << 20):
    """Queue device-to-device copies (enqueued without waiting) so stream 0 has work ahead of the uploads."""
    a, b = DeviceAllocator.malloc_cuda(nbytes), DeviceAllocator.malloc_cuda(nbytes)
    for _ in range(seconds_hint_copies):
        DeviceAllocator.memcpy(b, a, nbytes, 3)
    return a, b


def _release(*ptrs):
    DeviceAllocator.sync_device()
    for p in ptrs:
        DeviceAllocator.free_cuda(p)


needs_staging = pytest.mark.skipif(not S.capable(), reason="the backend keeps the synchronous upload (no page-locked "
                                   "ring: Metal's unified memory, Level Zero)")


def test_every_entry_the_staged_path_needs_is_in_both_c_abi_rows():
    for name in ("cuda", "hip"):
        missing = [k for k in staging.NEEDS if k not in _rt._GPU_BACKENDS[name]]
        assert not missing, f"{name} lacks {missing}"


@needs_staging
def test_an_array_changed_after_the_call_is_uploaded_as_it_was_at_the_call():
    bufs = _busy()
    staged0 = S.staged
    outs, wants = [], []
    host = np.zeros(37, dtype=np.int64)
    for i in range(64):
        host[:] = np.arange(37) * (i + 1)
        wants.append(host.copy())
        outs.append(NBXTensor.from_numpy(host))
        host[:] = -1                                     # the caller reuses its array at once
    _release(*bufs)
    assert S.staged - staged0 >= 1, "the queue was busy, yet nothing went through the ring"
    for t, w in zip(outs, wants):
        assert np.array_equal(t.numpy(), w)


@needs_staging
def test_the_host_does_not_wait_for_the_queue():
    bufs = _busy()
    t0 = time.perf_counter()
    for i in range(16):
        NBXTensor.from_numpy(np.full(32, i, dtype=np.int32))
    host_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    DeviceAllocator.sync_device()
    drain_s = time.perf_counter() - t1
    _release(*bufs)
    assert host_s < drain_s / 4, f"16 uploads took {host_s * 1e3:.2f} ms of host time with {drain_s * 1e3:.2f} ms " \
                                 f"of queue still ahead: they waited for it"


@needs_staging
def test_the_ring_grows_instead_of_overwriting_a_copy_in_flight():
    ring = S._rings.get(DeviceAllocator.get_device())
    count = (ring.cap // (1 << 15) if ring else 0) + 8   # more than the ring holds now, whatever earlier tests grew it to
    bufs = _busy()
    grown0 = S.grown
    arrays = [np.random.default_rng(i).integers(0, 1 << 30, size=1 << 12, dtype=np.int64) for i in range(count)]
    outs = [NBXTensor.from_numpy(a) for a in arrays]     # count x 32 KB, all still in flight
    _release(*bufs)
    assert S.grown > grown0
    for t, a in zip(outs, arrays):
        assert np.array_equal(t.numpy(), a)


def test_an_idle_queue_takes_the_direct_copy_and_grows_nothing():
    DeviceAllocator.sync_device()
    direct0, grown0 = S.direct, S.grown
    t = NBXTensor.from_numpy(np.arange(5, dtype=np.float32))
    assert S.direct == direct0 + 1 and S.grown == grown0
    assert np.array_equal(t.numpy(), np.arange(5, dtype=np.float32))


def test_the_interceptor_in_force_sees_the_upload_as_an_h2d_copy():
    seen = []

    class Count(AllocatorInterceptor):
        def memcpy(self, direct, dst, src, nbytes, kind):
            if kind == 1:
                seen.append((nbytes, bytes((ctypes.c_uint8 * nbytes).from_address(src))))
            return direct(dst, src, nbytes, kind)

    host = np.arange(3, dtype=np.int32)
    i = Count()
    DeviceAllocator.install_interceptor(i)
    try:
        t = NBXTensor.from_numpy(host)
    finally:
        DeviceAllocator.remove_interceptor(i)
    assert seen == [(12, host.tobytes())]
    assert np.array_equal(t.numpy(), host)
