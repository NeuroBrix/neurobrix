"""The drain's fast path must skip an empty queue and never a busy one.

`_nbx_queue_drain()` calls `runtime().sync()` only when
`has_pending_gpu_writes()` says this queue holds work that could be writing
memory. That predicate carries the whole saving and the whole risk:

  * if it wrongly says True, the optimisation quietly stops working and the
    decode goes back to paying 0.458 ms per launch with nobody noticing;
  * if it wrongly says False while a blit is in flight, a kernel reads memory
    our queue has not finished writing — which is the defect
    `test_the_two_queues_are_ordered`'s third crossing exists to catch.

Both directions are asserted here, so neither can rot silently.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("triton")

from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator  # noqa: E402


def _metal_or_skip():
    try:
        from neurobrix.kernels.nbx_tensor import _detect_gpu_backend
        if _detect_gpu_backend() != "metal":
            pytest.skip("this queue is Metal's; there is one queue elsewhere")
        from neurobrix.kernels.metal_device import runtime
        return runtime()
    except Exception:
        pytest.skip("no Metal runtime the engine can resolve")


N = 1 << 20          # 4 MB of int32 — big enough that the blit is still in flight


def test_an_idle_queue_reports_no_pending_writes():
    rt = _metal_or_skip()
    rt.sync()                                   # leave it provably idle
    assert rt.has_pending_gpu_writes() is False, (
        "an idle queue claims pending writes: the drain's fast path will never "
        "fire and the 74.9 % of launches that skip it will start paying again")


def test_a_queue_with_a_blit_in_flight_reports_pending_writes():
    rt = _metal_or_skip()
    rt.sync()
    src = NBXTensor.from_numpy(np.full(N, 0x3C3C3C3C, dtype=np.int32))
    dst = NBXTensor.from_numpy(np.zeros(N, dtype=np.int32))
    DeviceAllocator.memcpy(dst.data_ptr(), src.data_ptr(), N * 4, 3)   # 3 = D2D
    try:
        assert rt.has_pending_gpu_writes() is True, (
            "a queue with a D2D blit just enqueued claims nothing pending: the "
            "drain would be skipped and a kernel could read what the blit has "
            "not finished writing")
    finally:
        rt.sync()


def test_the_blit_is_visible_once_the_queue_reports_idle_again():
    """The predicate is about ORDERING, so the data must actually be there."""
    rt = _metal_or_skip()
    rt.sync()
    value = 0x3C3C3C3C
    src = NBXTensor.from_numpy(np.full(N, value, dtype=np.int32))
    dst = NBXTensor.from_numpy(np.zeros(N, dtype=np.int32))
    DeviceAllocator.memcpy(dst.data_ptr(), src.data_ptr(), N * 4, 3)
    rt.sync()
    assert rt.has_pending_gpu_writes() is False
    assert int((dst.numpy() != value).sum()) == 0
