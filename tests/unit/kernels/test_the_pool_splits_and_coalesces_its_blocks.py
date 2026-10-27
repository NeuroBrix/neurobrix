"""The Triton allocator's pool splits and coalesces (S4a, 2026-10-27): a request takes the smallest free block that
holds it and splits off the rest; a freed block merges with its free neighbours; only a wholly free segment returns to
the driver. Exact-size parking kept a freed 3 GB block from serving a 2 GB request (2.6-3.7 GB parked beside 6-7 GB
live, mochi-1-preview triton at the 8 GB rung). Card-less: the driver is a fake that hands out disjoint ranges.

What each test would do if the code were wrong: a freed 3 MB block not serving a 2 MB request (a second driver
malloc — red); two live blocks overlapping (red); a coalesced segment never returned over the cap (red); a block of a
segment handed to the driver alone (the fake refuses a pointer it never issued — red)."""
import ctypes
import random

import pytest

from neurobrix.kernels import nbx_tensor as N

DA = N.DeviceAllocator
MB = 1 << 20


class FakeRT:
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
        p = ptr.value
        assert p in self.live, f"the driver never issued {p:#x} as an allocation"
        del self.live[p]
        return 0


@pytest.fixture
def rt(monkeypatch):
    fake = FakeRT()
    monkeypatch.setattr(N, "_gpu_runtime", lambda: fake)
    monkeypatch.setattr(N, "_active_backend", lambda: {"malloc": "cudaMalloc", "free": "cudaFree"})
    monkeypatch.setattr(DA, "_pool_enabled_init", True, raising=False)
    for name, value in (("_pool_enabled", True), ("_blocks", {}), ("_pool_free", {}),
                        ("_pool_sizes", {}), ("_pool_cached_bytes", {}), ("_pool_cached_peak", {}),
                        ("_pool_alloc_size", {}), ("_cuda_ptr_size", {}), ("_cuda_ptr_device", {}),
                        ("_cuda_ptr_epoch", {}), ("_cuda_live_bytes", {}), ("_cuda_peak_bytes", {}),
                        ("_range_size", {}), ("_range_bases", []), ("_pool_cap_bytes", {0: 0})):
        monkeypatch.setattr(DA, name, value)
    monkeypatch.setattr(DA, "get_device", staticmethod(lambda: 0))
    return fake


def test_a_freed_large_block_serves_a_smaller_request_and_coalesces_back(rt):
    a = DA.malloc_cuda(3 * MB)
    DA.free_cuda(a)
    b = DA.malloc_cuda(2 * MB)
    assert rt.mallocs == 1 and b == a, "the freed 3 MB block did not serve the 2 MB request"
    c = DA.malloc_cuda(MB)
    assert rt.mallocs == 1 and c == a + 2 * MB, "the split-off remainder did not serve the next request"
    DA.free_cuda(b)
    DA.free_cuda(c)
    assert DA._blocks[a][0] == 3 * MB and DA._blocks[a][1] is None and DA._blocks[a][2] is None, "not coalesced"
    assert DA._pool_cached_bytes[0] == 3 * MB


def test_a_wholly_free_segment_over_the_cap_returns_to_the_driver(rt, monkeypatch):
    monkeypatch.setattr(DA, "_pool_cap_bytes", {0: 2 * MB})
    a = DA.malloc_cuda(3 * MB)
    b = DA.malloc_cuda(MB)
    DA.free_cuda(b)                           # 1 MB parked, under the cap
    assert len(rt.live) == 2
    DA.free_cuda(a)                           # 4 MB parked, over the cap: whole segments go, largest first
    assert DA._pool_cached_bytes[0] <= 2 * MB and len(rt.live) == 1


def test_live_blocks_never_overlap_under_random_traffic(rt):
    rng = random.Random(0)
    live = {}
    for _ in range(3000):
        if live and rng.random() < 0.45:
            p = rng.choice(list(live))
            DA.free_cuda(p)
            del live[p]
        else:
            n = rng.choice([512, 4096, MB, 3 * MB, 5 * MB + 7, 17 * MB])
            p = DA.malloc_cuda(n)
            assert p not in live
            live[p] = n
        spans = sorted((p, p + n) for p, n in live.items())
        assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1)), "two live blocks overlap"
    for p in list(live):
        DA.free_cuda(p)
    DA._pool_flush(0)
    assert rt.live == {} and DA._blocks == {}, "a segment stayed with the driver or a block stayed in the pool"
