"""A command buffer the Metal runtime opens for an event or a synchronize is released when the call returns.

`-[MTLCommandQueue commandBuffer]` returns an AUTORELEASED object, and a Python thread has no autorelease
pool: without one around the call, every event record and every synchronize left its command buffer in
the process for good. Measured 2026-10-11 on an M4 Pro: +3.1 KB per record/record/synchronize round
and +1.6 KB per bare `stream_synchronize`. `do_bench` times every candidate of a sweep by such events, so
a certifier walk grew ~5 MB per key (0.6 -> 11.9 GB in 2.5 h, every chatterbox pass killed by the
memory guard). The gauge is the process's physical footprint (`ri_phys_footprint`): a command buffer is
not a Metal allocation, so `currentAllocatedSize()` never sees it.
"""
from __future__ import annotations

import ctypes
import os

import pytest

from neurobrix.kernels import nbx_tensor
from neurobrix.kernels.nbx_tensor import DeviceAllocator, NBXDtype, NBXTensor


def _on_metal() -> bool:
    try:
        return nbx_tensor._detect_gpu_backend() == "metal"
    except Exception:
        return False


metal_only = pytest.mark.skipif(not _on_metal(), reason="needs an Apple GPU with unified memory")


def _phys_footprint() -> int:
    """`proc_pid_rusage(RUSAGE_INFO_V2).ri_phys_footprint`: what `footprint -f` reports."""
    class _RUsageV2(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"f{i}", ctypes.c_uint64) for i in range(7)] \
            + [("ri_phys_footprint", ctypes.c_uint64)] + [(f"g{i}", ctypes.c_uint64) for i in range(20)]
    info = _RUsageV2()
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    assert libc.proc_pid_rusage(os.getpid(), 2, ctypes.byref(info)) == 0
    return int(info.ri_phys_footprint)


def _growth_mib(step, rounds: int) -> float:
    for _ in range(500):                                       # warm: first-use allocations settle
        step()
    before = _phys_footprint()
    for _ in range(rounds):
        step()
    return (_phys_footprint() - before) / 2 ** 20


@metal_only
def test_timing_events_leave_nothing_behind():
    NBXTensor.empty((16,), dtype=NBXDtype.float32)

    def step():
        s = DeviceAllocator.create_event(timing=True); e = DeviceAllocator.create_event(timing=True)
        DeviceAllocator.record_event(s, 0); DeviceAllocator.record_event(e, 0)
        DeviceAllocator.event_synchronize(e); DeviceAllocator.event_elapsed_ms(s, e)
        DeviceAllocator.destroy_event(s); DeviceAllocator.destroy_event(e)
    grew = _growth_mib(step, 10_000)
    assert grew < 4, f"10 000 timed event pairs grew the footprint by {grew:.1f} MiB (leaked ~31 MiB)"


@metal_only
def test_a_synchronize_leaves_nothing_behind():
    NBXTensor.empty((16,), dtype=NBXDtype.float32)
    grew = _growth_mib(lambda: DeviceAllocator.stream_synchronize(0), 10_000)
    assert grew < 4, f"10 000 synchronizes grew the footprint by {grew:.1f} MiB (leaked ~16 MiB)"
