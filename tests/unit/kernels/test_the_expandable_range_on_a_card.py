"""R16 on a real card: the driver's VMM primitives as `memory.vmm.Vmm` calls them (reserve, create, map + access,
unmap, release), and the allocator's large class growing in an expandable range with the reclaim at a refusal — the
pinned remainder of `test_the_large_class_grows_in_an_expandable_range` on the device itself, under an imposed rung.

What each test would do if the code were wrong: a mapped chunk the device cannot write (red: the memset fails); the
10 MB request refused although the free remainder's chunks were whole (red); bytes written in a live block changed by
the reclaim (red). One card."""
from types import SimpleNamespace

import pytest

import nbxtensor.backend.runtime as _rt
import nbxtensor.ops.wrappers as _wrappers
from nbxtensor.memory.allocator import DeviceAllocator as DA
from nbxtensor.memory.vmm import Vmm

pytestmark = pytest.mark.usefixtures("needs_device")
MB = 1 << 20


def test_the_driver_primitives_map_a_chunk_the_device_can_write():
    lib = Vmm.load(_rt.backend_table().get("driver_libs"))
    assert lib is not None
    DA.set_device(0)
    v = Vmm(lib, 0)
    assert v.supported()
    rc, gran = v.granularity()
    assert rc == 0 and gran > 0
    rc, va = v.reserve(4 * gran)
    assert rc == 0
    rc, h = v.create(gran)
    assert rc == 0 and v.map(va, gran, h) == 0
    assert _rt.device_memory().memset(va, 7, gran) == 0
    DA.sync_direct()
    assert v.unmap(va, gran) == 0 and v.release(h) == 0 and v.unreserve(va, 4 * gran) == 0


def test_a_pinned_remainder_is_given_back_on_the_card(monkeypatch):
    import numpy as np
    DA.set_device(0)
    lib = Vmm.load(_rt.backend_table().get("driver_libs"))
    _rc, gran = Vmm(lib, 0).granularity()
    prof = SimpleNamespace(devices=[SimpleNamespace(index=0, memory_mb=1024, measured={"vmm_chunk_bytes": gran})])
    monkeypatch.setattr(_wrappers, "get_hardware_profile", lambda: prof)
    monkeypatch.setattr(DA, "_VMM", {})
    from nbxtensor.memory.pool import SegmentPool
    monkeypatch.setattr(DA, "_POOL", SegmentPool(on_block=DA._block_reported))
    DA.empty_cache_pool()
    base = DA._cuda_live_bytes.get(0, 0)
    monkeypatch.setattr(DA, "_rung_limit_bytes", {0: base + 20 * MB})
    x = DA.malloc_cuda(16 * MB)
    DA.free_cuda(x)
    a = DA.malloc_cuda(8 * MB)
    b = DA.malloc_cuda(8 * MB)
    assert DA._POOL.ranges.get(0), "the large class did not grow in a range"
    src = np.full(2 * MB, 3, dtype=np.float32)
    DA.memcpy(b, src.ctypes.data, 8 * MB, 1)
    DA.free_cuda(a)
    c = DA.malloc_cuda(10 * MB)
    assert c is not None
    out = np.empty(2 * MB, dtype=np.float32)
    DA.memcpy(out.ctypes.data, b, 8 * MB, 2)
    assert (out == 3).all(), "the reclaim touched a live block"
    DA.free_cuda(b)
    DA.free_cuda(c)
    DA.empty_cache_pool()
