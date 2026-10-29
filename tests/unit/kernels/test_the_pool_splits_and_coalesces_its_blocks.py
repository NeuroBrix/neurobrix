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

from nbxtensor import tensor as N
import nbxtensor.memory.allocator as _nbx_allocator
from nbxtensor.memory.pool import SegmentPool
import nbxtensor.backend.runtime as _nbx_runtime

DA = _nbx_allocator.DeviceAllocator
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
    monkeypatch.setattr(_nbx_runtime, "_gpu_runtime", lambda: fake)
    monkeypatch.setattr(_nbx_runtime, "_active_backend", lambda: {"malloc": "cudaMalloc", "free": "cudaFree"})
    monkeypatch.setattr(DA, "_pool_enabled_init", True, raising=False)
    for name, value in (("_pool_enabled", True), ("_POOL", SegmentPool(on_block=DA._block_reported)),
                        ("_cuda_ptr_size", {}), ("_cuda_ptr_device", {}),
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
    assert DA._POOL.blocks[a][0] == 3 * MB and DA._POOL.blocks[a][1] is None and DA._POOL.blocks[a][2] is None, "not coalesced"
    assert DA._POOL.cached[0] == 3 * MB


def test_a_wholly_free_segment_over_the_cap_returns_to_the_driver(rt, monkeypatch):
    monkeypatch.setattr(DA, "_pool_cap_bytes", {0: 2 * MB})
    a = DA.malloc_cuda(3 * MB)
    b = DA.malloc_cuda(MB)
    DA.free_cuda(b)                           # 1 MB parked, under the cap
    assert len(rt.live) == 2
    DA.free_cuda(a)                           # 4 MB parked, over the cap: whole segments go, largest first
    assert DA._POOL.cached[0] <= 2 * MB and len(rt.live) == 1


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
    assert rt.live == {} and DA._POOL.blocks == {}, "a segment stayed with the driver or a block stayed in the pool"


@pytest.mark.parametrize("cap_mb", [0, 8])
def test_the_simulation_holds_what_the_allocator_holds(rt, monkeypatch, cap_mb):
    """`memory.pool.simulate` — the pricing replay of an allocation sequence (M3) — runs the allocator's own pool:
    on random traffic, the bytes it holds from the driver at their peak are the bytes the allocator held from this
    fake driver, to the byte, with the parked cap unbounded and bounded, weight arenas outside the pool among them. What
    it would do if the two ran different rules: differ (red)."""
    from nbxtensor.memory.pool import simulate
    monkeypatch.setattr(DA, "_pool_cap_bytes", {0: cap_mb * MB})
    held_peak = [0]
    real_malloc = rt.cudaMalloc

    def counting_malloc(out, size):
        rc = real_malloc(out, size)
        held_peak[0] = max(held_peak[0], sum(rt.live.values()))
        return rc

    rt.cudaMalloc = counting_malloc
    rng = random.Random(cap_mb + 1)
    live, events = {}, []
    for i in range(3000):
        if live and rng.random() < 0.45:
            k = rng.choice(list(live))
            DA.free_cuda(live.pop(k))
            events.append(("free", k, 0))
        else:
            n = rng.choice([512, 4096, MB, 3 * MB, 5 * MB + 7, 17 * MB])
            arena = rng.random() < 0.1                  # a block outside the pool: a weight arena
            live[i] = DA.malloc_cuda(n, pooled=not arena)
            events.append(("arena" if arena else "alloc", i, n))
    sim = simulate(events, parked_cap=cap_mb * MB)
    assert sim.peak_bytes == held_peak[0], (sim, held_peak[0])
    assert sim.peak_live_bytes <= sim.peak_bytes


def test_a_window_holds_the_most_held_since_it_opened_and_leaves_the_peak_alone(rt, monkeypatch):
    """`DeviceAllocator.open_window` / `close_window`: what a component holds from the driver from its load to its
    release (the release line says it) — the most held while the window was open, not the process's peak, which no
    window resets. What it would do if wrong: the process peak or the bytes at the close instead (red)."""
    monkeypatch.setattr(DA, "_windows", {})
    monkeypatch.setattr(DA, "_window_starts", {})
    big = DA.malloc_cuda(20 * MB)
    DA.free_cuda(big)
    DA._pool_flush(0)                                  # the process peak stays 20 MB
    resident = DA.malloc_cuda(2 * MB)                   # held before the window: not its own
    t = DA.open_window(0)
    assert DA.window_start(t) == 2 * MB
    a = DA.malloc_cuda(3 * MB)
    b = DA.malloc_cuda(5 * MB)
    DA.free_cuda(a)
    DA.free_cuda(b)
    DA._pool_flush(0)
    assert DA.close_window(0, t) == 10 * MB
    assert DA._cuda_peak_bytes[0] == 20 * MB
    assert DA.close_window(0, t) is None and DA.window_start(t) is None, "a closed window answers no figure"
    DA.free_cuda(resident)
