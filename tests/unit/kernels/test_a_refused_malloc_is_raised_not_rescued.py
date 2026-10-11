"""A refused device allocation is RAISED, named by its op — never rescued by draining the
deferred-free queue and retrying (ZERO FALLBACK, 2026-10-11).

Until 2026-10-11 `DeviceAllocator.malloc_cuda` answered a refused malloc by calling the hot loops'
reclaim hooks (drain the deferred dead-tensor queue) and retrying: `[NBX_OOM_RECLAIM] freed ...`
fired 1 279 times in Allegro's 16 GB triton run of the mq24 gate, up to 1 883 MB a time in
CogVideoX-5b-I2V's. The plan had not priced the queue, and the run fitted only through the rescue.
Prism now prices it (`PrismSolver._deferred_drain_mb`); a malloc refused anyway is a plan defect
and is raised, and the hot loop names the op, what it asked, what the queue held and the price.

What these tests would do on the old code: the first one would PASS the retried malloc (the hook
"frees" and the fake driver then grants) and fail on `pytest.raises`; the second would fail on the
import of `_deferred_drain_context`, which did not exist.

CPU only: the driver is a fake that refuses the first malloc and grants every later one.
"""
from __future__ import annotations

import pytest

import neurobrix.kernels.nbx_tensor as nt
from neurobrix.kernels.nbx_tensor import DeviceAllocator, DeviceOOMError

_OOM = 2   # cudaErrorMemoryAllocation


class _RefuseOnceRuntime:
    """A driver that refuses the first allocation and grants every later one — exactly what a
    retry after a reclaim would have met."""

    def __init__(self):
        self.mallocs = 0

    def cudaMalloc(self, ptr_ref, size):
        self.mallocs += 1
        if self.mallocs == 1:
            return _OOM
        ptr_ref._obj.value = 0x10000
        return 0

    def cudaGetLastError(self):
        return 0

    def cudaMemGetInfo(self, free_ref, total_ref):
        free_ref._obj.value = 64 << 20
        total_ref._obj.value = 16 << 30
        return 0


def test_a_refused_malloc_raises_and_never_drains_the_deferred_queue(monkeypatch):
    rt = _RefuseOnceRuntime()
    monkeypatch.setattr(nt, "_gpu_runtime", lambda: rt)
    monkeypatch.setattr(nt, "_active_backend", lambda: {
        "malloc": "cudaMalloc", "get_last_error": "cudaGetLastError",
        "mem_get_info": "cudaMemGetInfo"})
    monkeypatch.setattr(DeviceAllocator, "_pool_enabled", False, raising=False)
    monkeypatch.setattr(DeviceAllocator, "_pool_enabled_init", True, raising=False)
    monkeypatch.delenv("NBX_GC_ON_OOM", raising=False)

    drained = []

    def queue_holding_2_gb(dev, nbytes):
        drained.append(nbytes)
        return 2 << 30          # what the old rescue would have reported freed

    monkeypatch.setattr(DeviceAllocator, "_oom_reclaim_hooks", [queue_holding_2_gb])
    with pytest.raises(DeviceOOMError) as info:
        DeviceAllocator.malloc_cuda(512 << 20, dev_idx=0)
    assert drained == [], "the allocator drained the deferred queue to rescue a refused malloc"
    assert rt.mallocs == 1, "the allocator retried a refused malloc"
    assert info.value.requested == 512 << 20
    assert "GPU malloc failed" in str(info.value)


def test_the_hot_loops_oom_line_states_the_ask_the_queue_and_its_price():
    from neurobrix.triton.sequence import _deferred_drain_context

    mb = 1024 * 1024
    oom = DeviceOOMError("GPU malloc failed (error 2) for 536870912 bytes", requested=512 * mb)
    inside = _deferred_drain_context(oom, 1605 * mb, 2_000_000_000)
    assert "asked 512 MB" in inside
    assert "holding 1,605 MB" in inside and "1,907 MB priced for it (the profile's drain cliff)" in inside
    assert "under-priced this op's live set" in inside
    over = _deferred_drain_context(oom, 2500 * mb, 2_000_000_000)
    assert "overran its price" in over
    # A run draining at the PLAN's cliff (Prism's room beside what it holds whole) names that one.
    planned = _deferred_drain_context(oom, 700 * mb, 904 * mb, True)
    assert "of the 904 MB priced for it (the plan's drain cliff)" in planned
    # Any other failure is passed through untouched: it must not read as a memory problem.
    assert _deferred_drain_context(RuntimeError("illegal memory access"), 1605 * mb, 2_000_000_000) == ""
