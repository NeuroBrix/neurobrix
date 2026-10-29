"""Allocator pool default-on gate pins (P-PREFILL-TRITON lever 1, 2026-09-02).

The free-list pool in `DeviceAllocator` cut the cold xlong triton prefill
by 41 % (one cudaMalloc + one cudaFree per op output, 82 s of host API
time on a 172 s run). Turning it on by default needed two more bricks,
found by the re-attribution run itself: the pool parked the LOAD phase's
temporaries, and the first attention chunk's autotune benchmark (a
256 MiB allocation the engine does not own) found 121 MiB free on a
32 GB card.

Pins (CPU-only — no device, no driver):
  1. Default semantics: the pool is on unless NBX_ALLOC_POOL=0.
  2. Observability: the free-list push / exact-hit / smallest-fit
     accounting (`_pool_cached_bytes`, peak, slack) matches the bytes
     moved, so NBX_ALLOC_STATS reads the pool's true contribution to the
     driver watermark.
  3. The autotune headroom guard retries a launch ONCE after an
     out-of-memory failure (pool flushed, reclaimers drained), re-raises
     any other error untouched, and gives up after a second OOM.
  4. The parked-bytes cap: a push that would cross the cap evicts the
     largest parked blocks first (driver free, live bytes decremented);
     a block larger than the whole cap is never parked.

Runnable two ways:
  - pytest:  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_alloc_pool_gate.py -v
  - script:  PYTHONPATH=src python3 tests/unit/kernels/test_alloc_pool_gate.py
"""
from __future__ import annotations

import os

try:
    import pytest
except ModuleNotFoundError:  # script-mode under the pytest-less GPU venv
    class _NoPytest:
        @staticmethod
        def raises(exc):
            class _Ctx:
                def __enter__(self):
                    return self

                def __exit__(self, et, ev, tb):
                    assert et is not None and issubclass(et, exc), f"expected {exc}"
                    return True
            return _Ctx()

    pytest = _NoPytest()  # type: ignore

from nbxtensor.memory.allocator import DeviceAllocator


class _PoolState:
    """Snapshot / restore of the class-level pool state around a test."""

    FIELDS = ("_POOL", "_pool_enabled", "_cuda_ptr_size", "_cuda_ptr_device",
              "_cuda_live_bytes", "_cuda_peak_bytes", "_pool_cap_bytes",
              "_range_size", "_range_bases", "_cuda_ptr_epoch")

    def __enter__(self):
        self.saved = {f: getattr(DeviceAllocator, f) for f in self.FIELDS}
        self.init = getattr(DeviceAllocator, "_pool_enabled_init", None)
        self.env = os.environ.get("NBX_ALLOC_POOL")
        for f in self.FIELDS:
            v = self.saved[f]
            setattr(DeviceAllocator, f, type(v)() if isinstance(v, dict) else v)
        from nbxtensor.memory.pool import SegmentPool
        DeviceAllocator._POOL = SegmentPool(on_block=DeviceAllocator._block_reported)   # a fresh pool, its stats at 0
        return self

    def __exit__(self, *a):
        for f in self.FIELDS:
            setattr(DeviceAllocator, f, self.saved[f])
        if self.init is None:
            if hasattr(DeviceAllocator, "_pool_enabled_init"):
                delattr(DeviceAllocator, "_pool_enabled_init")
        else:
            DeviceAllocator._pool_enabled_init = self.init
        if self.env is None:
            os.environ.pop("NBX_ALLOC_POOL", None)
        else:
            os.environ["NBX_ALLOC_POOL"] = self.env
        return False


def _reinit(env_value):
    if hasattr(DeviceAllocator, "_pool_enabled_init"):
        delattr(DeviceAllocator, "_pool_enabled_init")
    if env_value is None:
        os.environ.pop("NBX_ALLOC_POOL", None)
    else:
        os.environ["NBX_ALLOC_POOL"] = env_value
    DeviceAllocator._maybe_init_pool()
    return DeviceAllocator._pool_enabled


def test_pool_is_on_by_default_and_zero_disables_it():
    with _PoolState():
        assert _reinit(None) is True
        assert _reinit("1") is True
        assert _reinit("0") is False


MiB = 1 << 20


class _FakeDriver:
    """Disjoint ranges handed out and taken back, so the pool's segments are real ones (S4a, 2026-10-27: blocks are
    slices of driver segments; a pointer the driver never issued is refused)."""
    def __init__(self):
        self.next, self.live, self.freed = 1 << 32, {}, []

    def cudaMalloc(self, out, size):
        p = self.next
        self.next += size.value + (2 * MiB)
        self.live[p] = size.value
        out._obj.value = p
        return 0

    def cudaFree(self, ptr):
        assert ptr.value in self.live, f"the driver never issued {ptr.value:#x}"
        self.freed.append(self.live.pop(ptr.value))
        return 0


def _with_driver(monkeypatch):
    import nbxtensor.tensor as nt
    import nbxtensor.backend.runtime as _nbx_runtime
    drv = _FakeDriver()
    monkeypatch.setattr(_nbx_runtime, "gpu_runtime", lambda: drv)
    monkeypatch.setattr(_nbx_runtime, "backend_table", lambda: {"malloc": "cudaMalloc", "free": "cudaFree"})
    monkeypatch.setattr(DeviceAllocator, "get_device", staticmethod(lambda: 0))
    monkeypatch.setattr(DeviceAllocator, "_rung_limit_bytes", {0: None})
    return drv


def test_free_list_accounting_tracks_bytes_hits_and_splits(monkeypatch):
    with _PoolState():
        _reinit(None)
        drv = _with_driver(monkeypatch)
        dev = 0
        DeviceAllocator._pool_cap_bytes[dev] = 1 << 40
        ptrs = [DeviceAllocator.malloc_cuda(n) for n in (1 * MiB, 4 * MiB, 6 * MiB)]
        for p in ptrs:
            DeviceAllocator.free_cuda(p)
        assert DeviceAllocator._POOL.cached[dev] == 11 * MiB
        assert DeviceAllocator._POOL.cached_peak[dev] == 11 * MiB
        st = DeviceAllocator._POOL.stats
        # Exact hit.
        assert DeviceAllocator._POOL.take(dev, 4 * MiB) == ptrs[1]
        assert st["exact"] == 1 and DeviceAllocator._POOL.cached[dev] == 7 * MiB
        # Smallest block that holds it, split: 5 MiB from the 6 MiB segment, 1 MiB left free beside it.
        assert DeviceAllocator._POOL.take(dev, 5 * MiB) == ptrs[2]
        assert st["fit"] == 1 and st["slack_total"] == 0
        assert DeviceAllocator._POOL.alloc_size[ptrs[2]] == 5 * MiB
        assert DeviceAllocator._POOL.cached[dev] == 2 * MiB
        # Miss: no free block holds 3 MiB.
        misses = st["miss"]
        assert DeviceAllocator._POOL.take(dev, 3 * MiB) is None
        assert st["miss"] == misses + 1
        # Peak unchanged by takes; nothing went back to the driver.
        assert DeviceAllocator._POOL.cached_peak[dev] == 11 * MiB and drv.freed == []


def test_autotune_headroom_guard_retries_once_then_gives_up():
    from nbxtensor.ops import wrappers

    calls = {"launch": 0, "flush": 0, "hook": 0}
    saved = (DeviceAllocator.empty_cache_pool, DeviceAllocator.get_device,
             getattr(DeviceAllocator, "_oom_reclaim_hooks", None))
    try:
        DeviceAllocator.empty_cache_pool = staticmethod(lambda: calls.__setitem__("flush", calls["flush"] + 1) or 0)  # type: ignore
        DeviceAllocator.get_device = staticmethod(lambda: 0)  # type: ignore
        DeviceAllocator._oom_reclaim_hooks = [lambda dev, n: calls.__setitem__("hook", calls["hook"] + 1) or 0]

        def launch_oom_once(*a, **k):
            calls["launch"] += 1
            if calls["launch"] == 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate 256.00 MiB.")
            return "ok"
        assert wrappers._autotune_headroom_guard(launch_oom_once)(1, x=2) == "ok"
        assert calls == {"launch": 2, "flush": 1, "hook": 1}

        def launch_other(*a, **k):
            raise RuntimeError("illegal memory access")
        with pytest.raises(RuntimeError):
            wrappers._autotune_headroom_guard(launch_other)()
        assert calls["flush"] == 1  # no flush, no retry for a non-OOM error

        def launch_oom_always(*a, **k):
            raise RuntimeError("CUDA out of memory. Tried to allocate 256.00 MiB.")
        with pytest.raises(RuntimeError):
            wrappers._autotune_headroom_guard(launch_oom_always)()
        assert calls["flush"] == 2  # one flush, one retry, then the raise
    finally:
        DeviceAllocator.empty_cache_pool, DeviceAllocator.get_device = saved[0], saved[1]  # type: ignore
        if saved[2] is None:
            delattr(DeviceAllocator, "_oom_reclaim_hooks")
        else:
            DeviceAllocator._oom_reclaim_hooks = saved[2]


def test_parked_cap_evicts_whole_segments_largest_first_and_never_parks_an_over_cap_block(monkeypatch):
    with _PoolState():
        _reinit(None)
        drv = _with_driver(monkeypatch)
        dev = 0
        DeviceAllocator._pool_cap_bytes[dev] = 10 * MiB
        a, b, c = (DeviceAllocator.malloc_cuda(6 * MiB), DeviceAllocator.malloc_cuda(3 * MiB),
                   DeviceAllocator.malloc_cuda(2 * MiB))
        DeviceAllocator.free_cuda(a)
        DeviceAllocator.free_cuda(b)
        assert DeviceAllocator._POOL.cached[dev] == 9 * MiB and drv.freed == []
        # The 2 MiB segment freed would park 11 MiB > cap: the largest wholly free segment (6 MiB) leaves first.
        DeviceAllocator.free_cuda(c)
        assert drv.freed == [6 * MiB]
        assert DeviceAllocator._POOL.cached[dev] == 5 * MiB
        assert DeviceAllocator._POOL.stats["evictions"] == 1
        assert DeviceAllocator._POOL.stats["evicted_bytes"] == 6 * MiB
        assert DeviceAllocator._cuda_live_bytes[dev] == 5 * MiB     # the evicted segment left the driver
        # A segment larger than the whole cap is never parked: straight back to the driver. The device grows for it,
        # so the parked 3 + 2 MiB go back to the driver first (no parked block survives a device growth).
        DeviceAllocator.free_cuda(DeviceAllocator.malloc_cuda(20 * MiB))
        assert sorted(drv.freed[1:3]) == [2 * MiB, 3 * MiB] and drv.freed[0] == 6 * MiB and drv.freed[3] == 20 * MiB
        assert DeviceAllocator._POOL.cached[dev] == 0


def test_no_parked_block_survives_a_device_growth(monkeypatch):
    """A malloc the pool cannot serve grows the device; every parked block goes back to the driver first, so the device
    holds what is alive, not what is alive plus what was once freed and split (Sana_1600M_4Kpx triton on a V100-16GB,
    lot 2026-10-28: 4.5 GB parked in 12 split segments beside 9.4 GB alive refused a 2.96 GB convolution)."""
    with _PoolState():
        _reinit(None)
        drv = _with_driver(monkeypatch)
        dev = 0
        DeviceAllocator._pool_cap_bytes[dev] = 1 << 40
        a = DeviceAllocator.malloc_cuda(4 * MiB)
        DeviceAllocator.free_cuda(a)
        assert DeviceAllocator._POOL.cached[dev] == 4 * MiB and drv.freed == []
        DeviceAllocator.malloc_cuda(8 * MiB)                  # the pool cannot serve it: the device grows
        assert drv.freed == [4 * MiB] and DeviceAllocator._POOL.cached[dev] == 0


def test_a_fit_hands_out_less_than_the_split_threshold_of_slack(monkeypatch):
    """HAT-S at 2048 px on a 16 GB V100 (2026-10-22): its 2 304 MB conv output took the freed 4 096 MB mask's
    parked block whole, 1 792 MB held that no plan priced. Since S4a a block is split whenever what is left reaches
    `_POOL_SPLIT_MIN`, so a fit hands out less than that: the remainder stays free for the next request."""
    with _PoolState():
        _reinit(None)
        _with_driver(monkeypatch)
        dev = 0
        DeviceAllocator._pool_cap_bytes[dev] = 1 << 40
        big = DeviceAllocator.malloc_cuda(4 * MiB)
        DeviceAllocator.free_cuda(big)
        got = DeviceAllocator._POOL.take(dev, 2304 * 1024)               # 2.25 MiB of 4: 1.75 MiB split off
        assert got == big and DeviceAllocator._POOL.alloc_size[got] == 2304 * 1024
        assert DeviceAllocator._POOL.cached[dev] == 4 * MiB - 2304 * 1024
        small = DeviceAllocator.malloc_cuda(2560 * 1024)
        DeviceAllocator.free_cuda(small)
        got = DeviceAllocator._POOL.take(dev, 2 * MiB)                   # 0.5 MiB left: under the threshold
        assert DeviceAllocator._POOL.stats["slack_max"] < DeviceAllocator._POOL.split_min, DeviceAllocator._POOL.stats


if __name__ == "__main__":
    test_pool_is_on_by_default_and_zero_disables_it()
    test_autotune_headroom_guard_retries_once_then_gives_up()
    print("OK 4/4")
