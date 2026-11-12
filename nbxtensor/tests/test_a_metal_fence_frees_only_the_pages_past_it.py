"""`memory.pages.MetalStateFence`: the release fence of the Triton kernels' queue on Metal, read from the Apple backend
handed in; a page released under a fence returns to its pool only once that fence has completed. Pure bookkeeping: a
stand-in backend, no card. It was the engine's triton/state_fence.py until 2026-11-12.

What this test would do if the code were wrong: a backend without a fence accepted (pages released while a kernel may
still read them), a page freed before its fence completed, or never freed after — RED."""
import pytest

from nbxtensor.memory.pages import MetalStateFence, PagePool, PageTable


class _Backend:
    def __init__(self):
        self.committed, self.done = 0, 0

    def fence(self):
        self.committed += 1
        return self.committed

    def completed_fence(self):
        return self.done


def test_a_backend_without_a_fence_is_refused_by_name():
    with pytest.raises(RuntimeError, match="ZERO FALLBACK: the Apple backend has no fence"):
        MetalStateFence(object())


def test_a_page_comes_back_once_its_fence_completed():
    native = _Backend()
    fence = MetalStateFence(native)
    pool = PagePool(num_pages=1, tokens_per_page=16, async_release=True)
    t = PageTable(pool)
    t.reserve(16)
    at = fence.record()
    t.free(fence=at)
    assert fence.drain(pool) == 0 and pool.free_pages == 0          # the kernels that read it may still run
    native.done = at
    assert fence.completed() == at and fence.drain(pool) == 1 and pool.free_pages == 1
