"""R16 (the supervisor's GO 2026-11-02 01:30): on a device with virtual memory management, the pool's large class grows
in ONE reserved address range — chunks mapped as it grows — and a refusal unmaps every chunk lying whole inside a free
block before it retries, so a free remainder beside a live block gives its pages back (Wan2.1-I2V triton 16g: 1.1 GB of
48 MB remainders pinned in driver segments refused a 731 MB attention product).

Card-less: the driver and the VMM are fakes; the card is the imposed rung (`memory.set_limit`).

What each test would do if the code were wrong: the request refused though whole free chunks were mappable (red); a
chunk of a LIVE block unmapped (the fake refuses to unmap a chunk under a live block — red); a profile that states no
chunk silently planned with one (red); a device whose chunk is 0 (no VMM) growing a range (red)."""
from types import SimpleNamespace

import pytest

import nbxtensor.backend.runtime as _nbx_runtime
import nbxtensor.memory.allocator as _alloc
import nbxtensor.ops.wrappers as _wrappers
from nbxtensor.memory.pool import SegmentPool

DA = _alloc.DeviceAllocator
MB = 1 << 20
CHUNK = 2 * MB


class FakeRT:
    def __init__(self):
        self.next, self.live = 1 << 32, {}

    def cudaMalloc(self, out, size):
        p = self.next
        self.next += size.value + CHUNK
        self.live[p] = size.value
        out._obj.value = p
        return 0

    def cudaFree(self, ptr):
        self.live.pop(ptr.value)
        return 0

    def cudaDeviceSynchronize(self):
        return 0


class FakeVmm:
    chunk = CHUNK

    def __init__(self):
        self.mapped, self.handles, self.n = {}, set(), 0

    def reserve(self, nbytes):
        return 0, 1 << 44

    def create(self, nbytes):
        self.n += 1
        self.handles.add(self.n)
        return 0, self.n

    def map(self, va, nbytes, h):
        assert va % CHUNK == 0 and va not in self.mapped
        self.mapped[va] = h
        return 0

    def unmap(self, va, nbytes):
        for p, (size, _prev, _next, free, dev, _cls) in DA._POOL.blocks.items():
            assert free or not (p < va + nbytes and va < p + size), "a chunk under a LIVE block was unmapped"
        self.mapped.pop(va)
        return 0

    def release(self, h):
        self.handles.discard(h)
        return 0


@pytest.fixture
def card(monkeypatch):
    fake, vmm = FakeRT(), FakeVmm()
    monkeypatch.setattr(_nbx_runtime, "gpu_runtime", lambda: fake)
    monkeypatch.setattr(_nbx_runtime, "backend_table",
                        lambda: {"malloc": "cudaMalloc", "free": "cudaFree", "sync": "cudaDeviceSynchronize",
                                 "driver_libs": ["fake"]})
    monkeypatch.setattr(DA, "_pool_enabled_init", True, raising=False)
    for name, value in (("_pool_enabled", True), ("_POOL", SegmentPool(on_block=DA._block_reported)),
                        ("_cuda_ptr_size", {}), ("_cuda_ptr_device", {}), ("_cuda_ptr_epoch", {}),
                        ("_cuda_live_bytes", {}), ("_cuda_peak_bytes", {}), ("_range_size", {}), ("_range_bases", []),
                        ("_pool_cap_bytes", {0: 0}), ("_rung_limit_bytes", {0: None}), ("_VMM", {0: vmm})):
        monkeypatch.setattr(DA, name, value)
    monkeypatch.setattr(DA, "get_device", staticmethod(lambda: 0))
    monkeypatch.setattr(DA, "_past_budget", staticmethod(lambda dm, dev, n: None))
    prof = SimpleNamespace(devices=[SimpleNamespace(index=0, memory_mb=64, measured={"vmm_chunk_bytes": CHUNK})])
    monkeypatch.setattr(_wrappers, "get_hardware_profile", lambda: prof)
    return vmm


def _pinned_remainder():
    """One 16 MB growth carved in two (A, B), A freed: an 8 MB free remainder beside live B in ONE grown region."""
    x = DA.malloc_cuda(16 * MB)
    DA.free_cuda(x)
    a = DA.malloc_cuda(8 * MB)            # the half rule splits: the remainder (8 MB) is at most the request
    b = DA.malloc_cuda(8 * MB)
    DA.free_cuda(a)
    return a, b


def test_a_free_remainder_beside_a_live_block_gives_its_pages_back_at_a_refusal(card):
    """On a 20 MB card: 16 MB grown, 8 MB of it free beside 8 MB live; a 10 MB request. In the range, the free
    remainder's 4 whole chunks are unmapped at the refusal and the request maps in their place."""
    DA._rung_limit_bytes[0] = 20 * MB
    a, b = _pinned_remainder()
    c = DA.malloc_cuda(10 * MB)
    assert c is not None and DA._cuda_live_bytes[0] <= 20 * MB
    assert not DA._POOL.blocks[b][3], "B stayed live"


def test_the_same_remainder_in_a_driver_segment_refuses_the_request(card, monkeypatch):
    """The control: the device measured without VMM (chunk 0) keeps driver segments — the remainder is pinned by B in
    its segment, the refusal's flush returns nothing, the 10 MB request is refused (Wan2.1-I2V's failure, small)."""
    DA._VMM.clear()
    prof = SimpleNamespace(devices=[SimpleNamespace(index=0, memory_mb=64, measured={"vmm_chunk_bytes": 0})])
    monkeypatch.setattr(_wrappers, "get_hardware_profile", lambda: prof)
    DA._rung_limit_bytes[0] = 20 * MB
    _pinned_remainder()
    with pytest.raises(_alloc.DeviceOOMError):
        DA.malloc_cuda(10 * MB)


def test_a_request_the_card_cannot_hold_even_after_the_reclaim_is_refused_by_name(card):
    DA._rung_limit_bytes[0] = 12 * MB
    DA.malloc_cuda(8 * MB)
    with pytest.raises(_alloc.DeviceOOMError, match="GPU malloc failed"):
        DA.malloc_cuda(8 * MB)


def test_a_profile_without_the_measured_chunk_is_refused_by_name(card, monkeypatch):
    DA._VMM.clear()
    prof = SimpleNamespace(devices=[SimpleNamespace(index=0, memory_mb=64, measured={})])
    monkeypatch.setattr(_wrappers, "get_hardware_profile", lambda: prof)
    with pytest.raises(RuntimeError, match="vmm_chunk_bytes"):
        DA.malloc_cuda(8 * MB)


def test_a_device_measured_without_vmm_keeps_its_driver_segments(card, monkeypatch):
    DA._VMM.clear()
    prof = SimpleNamespace(devices=[SimpleNamespace(index=0, memory_mb=64, measured={"vmm_chunk_bytes": 0})])
    monkeypatch.setattr(_wrappers, "get_hardware_profile", lambda: prof)
    p = DA.malloc_cuda(8 * MB)
    assert not DA._POOL.ranges.get(0) and p in _nbx_runtime.gpu_runtime().live
