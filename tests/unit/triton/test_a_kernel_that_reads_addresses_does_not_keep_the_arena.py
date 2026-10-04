"""A kernel that reads through a captured address does not keep the arena it read from.

triton-ext names the exposed buffers to Metal when it launches a kernel compiled as reading addresses
(`reads_addresses`: `[exposedBuffers() allObjects]`, handed to the encoder's `useResources`). That array is an
autoreleased object, and a Python thread has no autorelease pool around the call: it is never released, and it
holds every exposed buffer. Ours is the pinned scope's wrap of a WHOLE allocation, so after one such launch the
arena stayed in the process for good, freed in the allocator's books and still in its footprint.

Measured on the Mac (M4 Pro, 2026-10-04, results/moe_pinned_scope_50e3a41e_2026_10_04/):
* a 2 GiB arena, a pinned view, one kernel reading through its address, the arena freed: 2 116 MB kept; the same
  launch inside an autorelease pool: 69-78 MB (what a first launch leaves anyway);
* deepseek-moe-16b-chat streamed, 4 segments "one resident at a time": the allocator held 10.0 then 10.6 GB, the
  footprint went 11.0 -> 20.3 GB while the second segment loaded, and the guard killed the run (swap 9.2 GB).

The driver now launches such a kernel inside its own autorelease pool. Red before: the arena stays.
"""
from __future__ import annotations

import ctypes
import gc
import os
import time

import numpy as np
import pytest

triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

from neurobrix.kernels.nbx_tensor import DeviceAllocator, NBXDtype, NBXTensor  # noqa: E402
from neurobrix.kernels.launcher import launch  # noqa: E402

ARENA_MB = 512
KEPT_MB = 128           # a quarter of the arena: a release that works leaves a few MB, a leak leaves all 512


def _metal_or_skip():
    try:
        from neurobrix.kernels.nbx_tensor import _detect_gpu_backend
        if _detect_gpu_backend() != "metal":
            pytest.skip("the autoreleased resident list is triton-ext's Metal launch; other drivers bind their own way")
        from neurobrix.triton import triton_ext_driver as drv
        return drv
    except Exception:
        pytest.skip("no Metal backend the engine can resolve")


class _RusageV2(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"f{i}", ctypes.c_uint64) for i in range(40)]


def _footprint_mb() -> int:
    """The process's phys_footprint (`proc_pid_rusage`, the 8th uint64 of rusage_info_v2; equal to
    `footprint -f bytes` on this process when the instrument was written). RSS does not see device buffers."""
    ru = _RusageV2()
    assert ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage(os.getpid(), 2, ctypes.byref(ru)) == 0
    return int(ru.f7 >> 20)


@triton.jit
def _read_through(tab_ptr, out_ptr, BLOCK: tl.constexpr):
    off = tl.arange(0, BLOCK)
    src = tl.cast(tl.load(tab_ptr), tl.pointer_type(tl.float32), bitcast=True)
    tl.store(out_ptr + off, tl.load(src + off) + 1.0)


def _arena(chunk):
    n = (ARENA_MB << 20) // 4
    t = NBXTensor.empty((n,), device="cuda", dtype=NBXDtype.float32)
    v = np.ctypeslib.as_array((ctypes.c_float * n).from_address(t.data_ptr()))
    for i in range(0, n, chunk.size):                  # incompressible: zeros would hide in the compressor
        v[i:i + chunk.size] = chunk[:min(chunk.size, n - i)]
    del v
    return t


def _read_one_view_through_its_address(drv, t, expect):
    view = t.narrow(0, 1024, 4096)
    out = NBXTensor.from_numpy(np.zeros(256, dtype=np.float32))
    with drv.pinned_addresses(view):
        tab = NBXTensor.from_numpy(np.array([drv.pinned_gpu_address(view.data_ptr())], dtype=np.int64))
        launch(_read_through, (1,), tab, out, BLOCK=256)
        DeviceAllocator.sync_device()
    assert abs(float(out.numpy()[0]) - (expect + 1.0)) < 1e-6, "the kernel did not read the view through its address"


def _settle():
    gc.collect()
    DeviceAllocator.sync_device()
    DeviceAllocator.empty_cache_pool()
    time.sleep(0.5)


def test_the_arena_leaves_the_footprint_after_a_kernel_read_through_its_address():
    drv = _metal_or_skip()
    chunk = np.random.default_rng(0).standard_normal(1 << 22, dtype=np.float32)

    warm = NBXTensor.from_numpy(chunk.copy())           # compile the kernel and pay a first launch's own residue
    _read_one_view_through_its_address(drv, warm, float(chunk[1024]))
    del warm
    _settle()

    before = _footprint_mb()
    t = _arena(chunk)
    held = _footprint_mb() - before
    assert held > ARENA_MB * 0.9, f"the instrument does not see the arena: {held} MB for {ARENA_MB}"
    _read_one_view_through_its_address(drv, t, float(chunk[1024]))
    del t
    _settle()
    kept = _footprint_mb() - before
    assert not drv._RESIDENT_WRAPS and not drv._PIN_COUNTS
    assert kept < KEPT_MB, (
        f"{kept} MB of a {ARENA_MB} MB arena are still in the footprint after it was freed: the launch that read "
        f"through its address left the exposed buffer retained")


def test_a_pinned_scope_alone_never_keeps_its_arena():
    """The same retention without any kernel, and only once in a while: reading the pinned wrap's address
    (`gpu_address()`) inserts it in triton-ext's weak table of exposed buffers, and an insert into a weak
    NSHashTable can load its live members, each load a retain + autorelease with no pool to drop it. Measured
    2026-10-04, 48 rounds on a fresh 128 MiB arena each (pinned scope, address read, arena freed): +128 MB kept
    at rounds 15, 30 and 45. One round cannot show it, so this runs enough rounds to cross the table's growth."""
    drv = _metal_or_skip()
    arena_mb, rounds = 64, 48
    chunk = np.random.default_rng(1).standard_normal((arena_mb << 20) // 4, dtype=np.float32)
    after = []
    for _ in range(rounds):
        t = NBXTensor.from_numpy(chunk)                # a fresh allocation, incompressible
        view = t.narrow(0, 1024, 4096)
        with drv.pinned_addresses(view):
            drv.pinned_gpu_address(view.data_ptr())
        del view, t
        gc.collect()
        DeviceAllocator.sync_device()
        DeviceAllocator.empty_cache_pool()
        after.append(_footprint_mb())
    kept = after[-1] - after[1]                         # from the second round: the first pays one-off costs
    assert kept < arena_mb, (
        f"{kept} MB kept over {rounds} pinned scopes on fresh {arena_mb} MB arenas ({kept / arena_mb:.1f} "
        f"arenas never returned); footprint after each round: {after}")
