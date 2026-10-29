"""Under an imposed rung (`NBX_PRISM_BUDGET_MB`) the Triton allocator holds to it as to a card of that size: a malloc
that would carry what the process holds from the driver (live and pooled) past the rung less the runtime's context is
refused as that card refuses it — the pool is flushed and the malloc retried, then a named OOM. Measured 2026-10-28:
mochi-1-preview triton at the 8 GB rung on a 32 GB card held 10.8 GB, 3.0-3.7 GB of it pooled free blocks no refused
malloc ever asked back. Card-less: the driver is a fake with 32 GB and a 300 MB context.

What each test would do if the code were wrong: a malloc past the rung served from the 32 GB card (no flush, the held
bytes over the rung — red); an OOM raised while the pool held enough to flush (red); no rung, a refusal (red)."""
import pytest

from nbxtensor import tensor as N
import nbxtensor.memory.allocator as _nbx_allocator
from nbxtensor.memory.pool import SegmentPool
import nbxtensor.backend.runtime as _nbx_runtime

DA = _nbx_allocator.DeviceAllocator
MB = 1 << 20


class FakeRT:
    """A 32 GB card whose runtime context holds 300 MB."""
    def __init__(self):
        self.next, self.live, self.mallocs = 1 << 32, {}, 0

    def cudaMalloc(self, out, size):
        p = self.next
        self.next += size.value + (1 << 21)
        self.live[p] = size.value
        self.mallocs += 1
        out._obj.value = p
        return 0

    def cudaFree(self, ptr):
        del self.live[ptr.value]
        return 0


@pytest.fixture
def rt(monkeypatch):
    fake = FakeRT()
    monkeypatch.setattr(_nbx_runtime, "_gpu_runtime", lambda: fake)
    monkeypatch.setattr(_nbx_runtime, "_active_backend", lambda: {"malloc": "cudaMalloc", "free": "cudaFree"})
    monkeypatch.setattr(DA, "_pool_enabled_init", True, raising=False)
    for name, value in (("_pool_enabled", True), ("_POOL", SegmentPool(on_block=DA._block_reported)),
                        ("_cuda_ptr_size", {}), ("_cuda_ptr_device", {}),
                        ("_cuda_ptr_epoch", {}), ("_cuda_live_bytes", {}), ("_cuda_peak_bytes", {}),
                        ("_range_size", {}), ("_range_bases", []), ("_pool_cap_bytes", {0: 1 << 40}),
                        ("_rung_limit_bytes", {})):
        monkeypatch.setattr(DA, name, value)
    monkeypatch.setattr(DA, "get_device", staticmethod(lambda: 0))
    monkeypatch.setattr(DA, "clear_last_error", staticmethod(lambda: None))
    monkeypatch.setattr(DA, "mem_get_info", staticmethod(
        lambda dev=None: (32768 * MB - 300 * MB - sum(fake.live.values()), 32768 * MB)))
    return fake


def test_a_malloc_past_the_rung_flushes_the_pool_and_the_held_bytes_stay_under_it(rt, monkeypatch):
    monkeypatch.setenv("NBX_PRISM_BUDGET_MB", "1024")          # this process may hold 1024 - 300 = 724 MB
    a = DA.malloc_cuda(400 * MB)
    DA.free_cuda(a)                                             # parked in the pool, still held from the driver
    b = DA.malloc_cuda(500 * MB)                                # 400 + 500 > 724: the pool is flushed first
    assert b and sum(rt.live.values()) == 500 * MB, f"held {sum(rt.live.values()) / MB:.0f} MB past the 724 MB rung"


def test_past_the_rung_with_nothing_to_flush_is_a_named_oom(rt, monkeypatch):
    monkeypatch.setenv("NBX_PRISM_BUDGET_MB", "1024")
    DA.malloc_cuda(600 * MB)
    with pytest.raises(_nbx_allocator.DeviceOOMError, match="imposed rung: 724MB"):
        DA.malloc_cuda(200 * MB)


def test_without_a_rung_the_card_is_the_physical_one(rt, monkeypatch):
    monkeypatch.delenv("NBX_PRISM_BUDGET_MB", raising=False)
    a = DA.malloc_cuda(600 * MB)
    DA.free_cuda(a)
    DA.malloc_cuda(2000 * MB)
    assert sum(rt.live.values()) == 2600 * MB, "the pool was flushed with no rung imposed"


def test_a_weight_arena_freed_goes_back_to_the_driver_whole_not_into_the_pool(rt, monkeypatch):
    """The 8 GB failure itself: a segment's weights freed into the pool, a small tensor that outlives it splits the
    freed block, and the next segment's arena finds the rung held by a pool it cannot flush. Unpooled, the freed arena
    leaves, and the small tensor takes its own segment."""
    monkeypatch.setenv("NBX_PRISM_BUDGET_MB", "1024")          # 724 MB this process may hold
    arena = DA.malloc_cuda(500 * MB, pooled=False)              # segment 0's weights
    DA.free_cuda(arena)                                         # released after segment 0
    assert not rt.live, "the freed weight arena stayed held from the driver"
    small = DA.malloc_cuda(19 * MB)                             # the encoder's output, outliving the segment
    DA.malloc_cuda(600 * MB, pooled=False)                      # the next segment's weights: 19 + 600 <= 724
    assert DA._POOL.blocks.get(small) is not None and sum(rt.live.values()) == 619 * MB
