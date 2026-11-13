"""nbxtensor.memory.deferred_frees.DeferredFrees.finish: a run's end frees its deferred tensors and synchronises first
only where a mid-run release would — several devices, or the allocator's pool off. On one device under the pool the
replayed decode step had recorded the end's synchronisation as a full device wait at every token.

What this test does if the code were wrong: the one-device cell fails when finish() synchronises under the pool; the
other cells fail when it does not synchronise across devices or with the pool off, or when the batch is not freed.
"""
from types import SimpleNamespace

import pytest

from nbxtensor.memory import deferred_frees as DF


class _T(SimpleNamespace):
    def nbytes(self):
        return 1024


@pytest.fixture
def synced(monkeypatch):
    calls = []
    monkeypatch.setattr(DF, "sync_devices_of", lambda ts: calls.append(len(list(ts))))
    return calls


@pytest.mark.parametrize("one_device, pooled, waits", [(True, True, False), (False, True, True), (True, False, True)])
def test_finish_waits_only_where_a_mid_run_release_waits(monkeypatch, synced, one_device, pooled, waits):
    monkeypatch.setattr(DF.DeviceAllocator, "pool_enabled", staticmethod(lambda: pooled))
    frees = DF.DeferredFrees(1 << 30, 1 << 20, one_device=one_device)
    frees.add(_T(device_idx=0))
    frees.add(_T(device_idx=0))
    assert frees.finish() == 2048 and len(frees) == 0
    assert synced == ([2] if waits else [])


def test_release_keeps_its_explicit_wait(synced):
    frees = DF.DeferredFrees(1 << 30, 1 << 20, one_device=True)
    frees.add(_T(device_idx=0))
    frees.release()
    assert synced == [1]
