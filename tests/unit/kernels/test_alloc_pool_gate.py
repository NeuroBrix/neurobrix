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

from neurobrix.kernels.nbx_tensor import DeviceAllocator


class _PoolState:
    """Snapshot / restore of the class-level pool state around a test."""

    FIELDS = ("_pool_free", "_pool_alloc_size", "_pool_enabled", "_pool_cached_bytes",
              "_pool_cached_peak", "_pool_stats", "_cuda_ptr_size", "_cuda_ptr_device",
              "_cuda_live_bytes", "_cuda_peak_bytes", "_pool_cap_bytes")

    def __enter__(self):
        self.saved = {f: getattr(DeviceAllocator, f) for f in self.FIELDS}
        self.init = getattr(DeviceAllocator, "_pool_enabled_init", None)
        self.env = os.environ.get("NBX_ALLOC_POOL")
        for f in self.FIELDS:
            v = self.saved[f]
            setattr(DeviceAllocator, f, type(v)() if isinstance(v, dict) else v)
        DeviceAllocator._pool_stats = {"exact": 0, "fit": 0, "miss": 0, "slack_total": 0,
                                       "slack_max": 0, "flushes": 0, "flushed_bytes": 0,
                                       "evictions": 0, "evicted_bytes": 0}
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


def test_free_list_accounting_tracks_bytes_hits_and_slack():
    with _PoolState():
        _reinit(None)
        dev = 0
        # Three blocks "allocated" (bookkeeping only) and returned to the pool.
        for ptr, nbytes in ((0x1000, 1024), (0x2000, 4096), (0x3000, 6000)):
            DeviceAllocator._cuda_ptr_size[ptr] = nbytes
            DeviceAllocator._cuda_ptr_device[ptr] = dev
            DeviceAllocator.free_cuda(ptr)
        assert DeviceAllocator._pool_cached_bytes[dev] == 1024 + 4096 + 6000
        assert DeviceAllocator._pool_cached_peak[dev] == 1024 + 4096 + 6000
        # Exact hit.
        assert DeviceAllocator._pool_take(dev, 4096) == 0x2000
        st = DeviceAllocator._pool_stats
        assert st["exact"] == 1 and DeviceAllocator._pool_cached_bytes[dev] == 1024 + 6000
        # Smallest-fit (5000 -> the 6000 block; slack 1000).
        assert DeviceAllocator._pool_take(dev, 5000) == 0x3000
        assert st["fit"] == 1 and st["slack_total"] == 1000 and st["slack_max"] == 1000
        assert DeviceAllocator._pool_alloc_size[0x3000] == 6000
        assert DeviceAllocator._pool_cached_bytes[dev] == 1024
        # Miss: nothing >= 3000 within 2x.
        assert DeviceAllocator._pool_take(dev, 3000) is None
        assert st["miss"] == 1
        # Peak unchanged by takes.
        assert DeviceAllocator._pool_cached_peak[dev] == 1024 + 4096 + 6000


def test_autotune_headroom_guard_retries_once_then_gives_up():
    from neurobrix.kernels import wrappers

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


def test_parked_cap_evicts_largest_first_and_never_parks_an_over_cap_block():
    import neurobrix.kernels.nbx_tensor as nt

    freed = []

    class _RT:
        def cudaFree(self, ptr):
            freed.append(ptr.value)
            return 0

    saved = (nt._gpu_runtime, nt._active_backend)
    try:
        nt._gpu_runtime = lambda: _RT()  # type: ignore
        nt._active_backend = lambda: {"free": "cudaFree"}  # type: ignore
        with _PoolState():
            _reinit(None)
            dev = 0
            DeviceAllocator._pool_cap_bytes[dev] = 10_000  # pinned: no profile in a bare process
            for ptr, nbytes in ((0x1000, 6000), (0x2000, 3000)):
                DeviceAllocator._cuda_ptr_size[ptr] = nbytes
                DeviceAllocator._cuda_ptr_device[ptr] = dev
                DeviceAllocator._cuda_live_bytes[dev] = DeviceAllocator._cuda_live_bytes.get(dev, 0) + nbytes
                DeviceAllocator.free_cuda(ptr)
            assert DeviceAllocator._pool_cached_bytes[dev] == 9000 and freed == []
            # A 2000-byte push would park 11000 > cap → evict the largest (6000) first.
            DeviceAllocator._cuda_ptr_size[0x3000] = 2000
            DeviceAllocator._cuda_ptr_device[0x3000] = dev
            DeviceAllocator._cuda_live_bytes[dev] += 2000
            DeviceAllocator.free_cuda(0x3000)
            assert freed == [0x1000]
            assert DeviceAllocator._pool_cached_bytes[dev] == 5000  # 3000 + 2000 parked
            assert DeviceAllocator._pool_stats["evictions"] == 1
            assert DeviceAllocator._pool_stats["evicted_bytes"] == 6000
            assert DeviceAllocator._cuda_live_bytes[dev] == 5000  # the evicted block left the driver
            assert 0x1000 not in DeviceAllocator._cuda_ptr_size
            # A block larger than the whole cap is never parked: straight to the driver.
            DeviceAllocator._cuda_ptr_size[0x4000] = 20_000
            DeviceAllocator._cuda_ptr_device[0x4000] = dev
            DeviceAllocator._cuda_live_bytes[dev] += 20_000
            DeviceAllocator.free_cuda(0x4000)
            assert freed == [0x1000, 0x4000]
            assert DeviceAllocator._pool_cached_bytes[dev] == 5000
    finally:
        nt._gpu_runtime, nt._active_backend = saved  # type: ignore


if __name__ == "__main__":
    test_pool_is_on_by_default_and_zero_disables_it()
    test_free_list_accounting_tracks_bytes_hits_and_slack()
    test_autotune_headroom_guard_retries_once_then_gives_up()
    test_parked_cap_evicts_largest_first_and_never_parks_an_over_cap_block()
    print("OK 4/4")
