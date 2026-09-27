"""A Metal object the runtime creates through a `new*` selector is released when the runtime frees it.

pyobjc up to 12.2.2 over-retains the result of instance `new*` methods (ronaldoussoren/pyobjc#690),
so every MTLBuffer the Metal runtime made outlived its free. Measured 2026-09-26 on an M4 Pro: after
`del` + gc + `empty_cache_pool()` of a written 2 GiB NBXTensor the allocator reported live 0 and
pool 0 while the process footprint stayed 2.4 GB; `MTLDevice.currentAllocatedSize()` never came
down. The gauge here is Metal's own allocated size, which counts objects that exist — not the
footprint, whose pages a purgeable or compressed buffer can hide.
"""
from __future__ import annotations

import gc

import numpy as np
import pytest

from neurobrix.kernels import metal_device, nbx_tensor
from neurobrix.kernels.nbx_tensor import DeviceAllocator, NBXDtype, NBXTensor


def _on_metal() -> bool:
    try:
        return nbx_tensor._detect_gpu_backend() == "metal"
    except Exception:
        return False


metal_only = pytest.mark.skipif(not _on_metal(), reason="needs an Apple GPU with unified memory")


@metal_only
def test_a_new_buffer_is_owned_once():
    NBXTensor.empty((16,), dtype=NBXDtype.float32)          # the runtime exists and has declared
    import Metal
    dev = Metal.MTLCreateSystemDefaultDevice()
    buf = dev.newBufferWithLength_options_(1 << 20, Metal.MTLResourceStorageModeShared)
    assert buf.retainCount() == 1, (
        f"a new MTLBuffer carries retainCount {buf.retainCount()}: pyobjc retained a +1 result "
        f"again and it will never be deallocated (pyobjc#690)")


@metal_only
def test_freed_device_memory_leaves_metal_including_after_gpu_work():
    import Metal
    dev = Metal.MTLCreateSystemDefaultDevice()
    mib = lambda: dev.currentAllocatedSize() / 2 ** 20
    src = NBXTensor.from_numpy(np.random.default_rng(26).standard_normal(1 << 20, dtype=np.float32))
    DeviceAllocator.empty_cache_pool(); gc.collect()
    before = mib()
    for _ in range(3):                                         # a blit makes the GPU hold the buffer
        t = NBXTensor.empty((64 << 20,), dtype=NBXDtype.float32)   # 256 MiB
        for k in range(64):
            DeviceAllocator.memcpy(t.data_ptr() + k * src._nbytes, src.data_ptr(), src._nbytes, kind=3)
        DeviceAllocator.stream_synchronize(0)
        del t
        gc.collect()
        DeviceAllocator.empty_cache_pool()
    after = mib()
    assert after - before < 32, (
        f"three 256 MiB tensors written by the GPU and freed left Metal's allocated size at "
        f"{before:.0f} -> {after:.0f} MiB: freed buffers are not being deallocated")
