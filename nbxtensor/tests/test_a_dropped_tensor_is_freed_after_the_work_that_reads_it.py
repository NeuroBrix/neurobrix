"""`nbxtensor.memory.deferred_frees`: the tensors a run drops wait in a `DeferredFrees` until a threshold releases
them — the bytes cliff or the count; earlier, with the pool off, when the device's free memory falls under a reserve —
every device owning one of them synchronised first, and the batch answers the allocator's out-of-memory reclaim while
it lives. The NeuroBrix engine's two hot loops held this queue themselves until 2026-11-12.

What this test would do if the code were wrong: a batch released under its cliff and count, one kept past them, the
pressure reading taken under the pool or more often than its interval, a release freeing before the devices
synchronised, a multi-device release mid-run without a synchronisation, a reclaimer left registered once the batch is
gone, the bytes counted wrong — RED."""
import gc

import numpy as np
import pytest

from nbxtensor.memory import deferred_frees as D
from nbxtensor.memory.allocator import DeviceAllocator as DA
from nbxtensor.tensor import NBXTensor


def _accelerator():
    if DA.device_count() == 0:          # only a machine without an accelerator skips; any other error fails
        pytest.skip("no accelerator on this machine")


def _t(n):
    return NBXTensor.from_numpy(np.zeros(n, np.float32))


@pytest.fixture
def synced(monkeypatch):
    seen = []
    monkeypatch.setattr(D, "sync_devices_of", lambda ts: seen.append(len(list(ts))))
    return seen


def test_the_cliff_and_the_count_release_the_batch(synced, monkeypatch):
    _accelerator()
    monkeypatch.setattr(DA, "pool_enabled", classmethod(lambda cls: True))
    frees = D.DeferredFrees(cliff=100, count=3)
    frees.add(_t(10))                                   # 40 bytes
    frees.add(None)
    assert (len(frees), frees.nbytes) == (1, 40) and frees.step(0) is None
    frees.add(_t(15))                                   # 100 bytes: the cliff
    assert frees.step(1) == D.Drain("cliff", 2, 100) and (len(frees), frees.nbytes) == (0, 0)
    assert synced == [], "one device under the pool: a mid-run release synchronises nothing"
    for _ in range(3):
        frees.add(_t(1))
    assert frees.step(2) == D.Drain("cliff", 3, 12)
    frees.add(_t(1))
    assert frees.release() == 4 and synced == [1], "the final release synchronises first"
    assert frees.release() == 0 and synced == [1], "an empty batch synchronises nothing"


def test_several_devices_or_no_pool_synchronise_every_mid_run_release(synced, monkeypatch):
    _accelerator()
    monkeypatch.setattr(DA, "pool_enabled", classmethod(lambda cls: True))
    frees = D.DeferredFrees(cliff=4, count=99, one_device=False)
    frees.add(_t(1))
    assert frees.step(0).reason == "cliff" and synced == [1]
    monkeypatch.setattr(DA, "pool_enabled", classmethod(lambda cls: False))
    frees = D.DeferredFrees(cliff=4, count=99)
    frees.add(_t(1))
    frees.step(0)
    assert synced == [1, 1]


def test_the_pressure_reading_is_throttled_and_never_taken_under_the_pool(synced, monkeypatch):
    _accelerator()
    reads = []
    free = {"bytes": 10_000}
    monkeypatch.setattr(DA, "device_free_bytes", staticmethod(lambda *a: reads.append(1) or free["bytes"]))
    monkeypatch.setattr(DA, "pool_enabled", classmethod(lambda cls: True))
    pooled = D.DeferredFrees(cliff=1 << 20, count=99, floor=4, reserve=1 << 30, interval=1)
    pooled.add(_t(4))
    assert pooled.step(0) is None and reads == []
    monkeypatch.setattr(DA, "pool_enabled", classmethod(lambda cls: False))
    frees = D.DeferredFrees(cliff=1 << 20, count=99, floor=32, reserve=5_000, interval=3)
    frees.add(_t(4))                                    # 16 bytes: under the floor
    assert frees.step(0) is None and reads == []
    frees.add(_t(4))                                    # 32 bytes: the floor; 10 000 free is over the reserve
    assert frees.step(1) is None and reads == [1]
    free["bytes"] = 100
    assert frees.step(2) is None and reads == [1], "read again before its interval"
    assert frees.step(4) == D.Drain("pressure", 2, 32) and reads == [1, 1] and synced == [2]


def test_a_live_batch_is_the_allocator_s_reclaimer_and_leaves_with_it(synced):
    _accelerator()
    gc.collect()                                        # the batches of the tests before leave first
    before = len(DA._oom_reclaim_hooks)
    frees = D.DeferredFrees(cliff=1 << 30, count=99)
    frees.add(_t(8))
    assert len(DA._oom_reclaim_hooks) == before + 1
    assert DA._oom_reclaim_hooks[-1](DA.get_device(), 0) == 32 and len(frees) == 0 and synced == [1]
    del frees
    gc.collect()
    assert len(DA._oom_reclaim_hooks) == before


def test_every_owning_device_is_synchronised_and_the_current_one_restored(monkeypatch):
    calls = []
    monkeypatch.setattr(DA, "get_device", staticmethod(lambda: 7))
    monkeypatch.setattr(DA, "set_device", staticmethod(lambda d: calls.append(("set", d))))
    monkeypatch.setattr(DA, "sync_device", staticmethod(lambda: calls.append(("sync",))))

    class On:
        def __init__(self, d):
            self.device_idx = d

    D.sync_devices_of([On(2), On(0), On(2)])
    assert calls == [("set", 0), ("sync",), ("set", 2), ("sync",), ("set", 7)]
    calls.clear()
    D.sync_devices_of([object()])
    assert calls == [("sync",)], "a batch naming no device synchronises the current one"


def test_the_queue_takes_every_slot_value_the_hot_loop_drops_lazy_proxies_included():
    """The hot loop drops whatever a slot held: an NBXTensor, a QuantizedTensor or a fused op's lazy proxy (a broadcast
    clone, a residual chain). The queue sizes each by `nbytes()`; a broadcast-clone proxy had only `_nbytes` and the run
    failed on CUDA (Sana 512 --triton, 2026-11-12). Every one now answers `nbytes()`."""
    from nbxtensor.ops import fused_upsample_conv as F, residual_chain as R
    from nbxtensor.quant.quantized_tensor import QuantizedTensor
    lazy = [c for m in (F, R) for c in vars(m).values() if isinstance(c, type) and "_nbytes" in vars(c)]
    for cls in lazy + [QuantizedTensor]:
        assert callable(getattr(cls, "nbytes", None)), f"{cls.__name__} has _nbytes but no nbytes()"
    proxy = F.BroadcastClonePyroxy.__new__(F.BroadcastClonePyroxy)
    for name in F.BroadcastClonePyroxy.__slots__:
        object.__setattr__(proxy, name, None)
    frees = D.DeferredFrees(cliff=10 ** 12, count=10 ** 6)
    frees.add(proxy)                                            # its _nbytes is 0 by design: it owns no storage
    assert frees.nbytes == proxy._nbytes
