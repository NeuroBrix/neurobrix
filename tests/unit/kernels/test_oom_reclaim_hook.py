"""Unit tests — DeviceAllocator OOM-reclaim hooks (P-TRITON-LIVE-SET).

The deferred-free drain policy's pressure gate compares device-free memory
to a static reserve and is blind to the SIZE of the incoming request: the
Sana 4Kpx triton VAE decode issued an 8 GiB malloc with 7.4 GiB free (above
the 6 GiB reserve, so no pressure drain) and failed by ~0.8 GiB while the
deferred dead-tensor queue legally held up to the 2 GB cliff of
reclaimable bytes. `DeviceAllocator.malloc_cuda` now calls registered
OOM-reclaim hooks after a failed device malloc and retries once if they
freed anything — the only chokepoint that sees both the request and the
reclaimable queue. These tests cover the registry mechanics (CPU) and the
end-to-end reclaim-then-retry on a real device (GPU, skipped without one).

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

from neurobrix.kernels.nbx_tensor import DeviceAllocator, NBXTensor


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
def test_oom_reclaim_drains_queue_and_retry_succeeds():
    """Fill the device, park 8 chunks (2 GiB) in a dead queue, then request
    an allocation larger than driver-free: the hook must drain the queue
    and the retried malloc must succeed."""
    held = []
    try:
        # Fill to capacity (natural OOM stops the fill — no hooks yet).
        while True:
            try:
                held.append(NBXTensor.empty((CHUNK // 4,), "float32"))
            except RuntimeError:
                break
        assert len(held) >= 10, "device too small for this test"

        # Free one chunk of true headroom, park 8 in the dead queue.
        held.pop()
        queue = [held.pop() for _ in range(8)]  # 2 GiB reclaimable
        drained = {"n": 0}

        def hook(dev, nbytes):
            if not queue:
                return 0
            DeviceAllocator.sync_device()
            freed = sum(t._nbytes for t in queue)
            queue.clear()
            drained["n"] += 1
            return freed

        DeviceAllocator.register_oom_reclaim_hook(hook)
        try:
            # ~1.5 GiB request: > the ~256 MiB free, < free + queue.
            big = NBXTensor.empty((6 * CHUNK // 4,), "float32")
            assert big is not None
            assert drained["n"] == 1, "hook did not fire exactly once"
            assert not queue, "queue not drained"
            del big
        finally:
            DeviceAllocator.unregister_oom_reclaim_hook(hook)
    finally:
        held.clear()


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
        test_oom_reclaim_drains_queue_and_retry_succeeds()
        test_oom_still_raises_when_nothing_reclaimable()
        print("ALL OOM-RECLAIM TESTS PASS (CPU+GPU)")
    else:
        print("CPU tests pass; GPU tests skipped (no device)")
