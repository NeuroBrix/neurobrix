"""Unit tests — DeviceAllocator deferred-queue reclaim hooks (P-TRITON-LIVE-SET).

The triton hot loops register their deferred dead-tensor queues here. Since
2026-10-11 `DeviceAllocator.malloc_cuda` never calls them — Prism prices the
queue and a refused malloc is raised
(`test_a_refused_malloc_is_raised_not_rescued.py`); the one reader left is the
autotune headroom guard (`test_alloc_pool_gate.py`). These tests cover the
registry mechanics (CPU) and that an impossible request raises (GPU, skipped
without one).

Runnable two ways:
  - pytest:  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_oom_reclaim_hook.py -v
  - script:  PYTHONPATH=src python3 tests/unit/kernels/test_oom_reclaim_hook.py
"""
from __future__ import annotations

try:
    import pytest
except ModuleNotFoundError:  # script-mode under a pytest-less GPU venv
    class _NoPytest:
        class mark:
            @staticmethod
            def skipif(*a, **k):
                return lambda fn: fn

        @staticmethod
        def skip(reason=""):
            raise SystemExit(f"SKIP: {reason}")

    pytest = _NoPytest()  # type: ignore

from nbxtensor.tensor import DeviceAllocator, NBXTensor


def _gpu_available() -> bool:
    try:
        t = NBXTensor.empty((8,), "float32")
        del t
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- CPU tests

def test_register_is_deduplicated():
    def hook(dev, nbytes):
        return 0

    before = len(DeviceAllocator._oom_reclaim_hooks)
    DeviceAllocator.register_oom_reclaim_hook(hook)
    DeviceAllocator.register_oom_reclaim_hook(hook)
    assert len(DeviceAllocator._oom_reclaim_hooks) == before + 1
    DeviceAllocator.unregister_oom_reclaim_hook(hook)
    assert len(DeviceAllocator._oom_reclaim_hooks) == before


def test_unregister_unknown_hook_is_noop():
    def hook(dev, nbytes):
        return 0

    before = list(DeviceAllocator._oom_reclaim_hooks)
    DeviceAllocator.unregister_oom_reclaim_hook(hook)  # never registered
    assert DeviceAllocator._oom_reclaim_hooks == before


# ---------------------------------------------------------------- GPU tests

CHUNK = 256 * 1024 * 1024  # 256 MiB fp32 chunks


@pytest.mark.skipif(not _gpu_available(), reason="no GPU")
def test_oom_still_raises_when_nothing_reclaimable():
    """With no hooks (or empty queues), an impossible request still raises
    the diagnostic RuntimeError — the reclaim path never masks a real OOM."""
    free = DeviceAllocator.device_free_bytes()
    assert free > 0
    try:
        NBXTensor.empty(((free + 4 * CHUNK) // 4,), "float32")
        raise AssertionError("expected RuntimeError OOM")
    except RuntimeError as e:
        assert "GPU malloc failed" in str(e)


if __name__ == "__main__":
    test_register_is_deduplicated()
    test_unregister_unknown_hook_is_noop()
    if _gpu_available():
        test_oom_still_raises_when_nothing_reclaimable()
        print("ALL OOM-RECLAIM TESTS PASS (CPU+GPU)")
    else:
        print("CPU tests pass; GPU tests skipped (no device)")
