"""`nbxtensor.backend.device_names`: a device name read once for every backend (`accelerator_index`), a tensor
moved to the device a name names (`to_named_device`), a named device waited for (`synchronize_named`). They lived in
the engine's Triton strategies (NeuroBrix `core/strategies/triton/base.py`, `value_parking.py`) until 2026-11-12.

What this test would do if the code were wrong: an AMD, Apple or Intel name read as no accelerator (the engine's old
`startswith("cuda")`), an ordinal dropped, a compound placement moved somewhere, the host name moving nothing, the
current device left on the waited one or not restored when the wait raises — RED."""
import numpy as np
import pytest

from nbxtensor.backend import device_names as M
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.tensor import NBXTensor


def test_every_backend_s_name_is_an_accelerator_and_keeps_its_ordinal():
    assert [M.accelerator_index(d) for d in ("cuda:1", "hip:2", "mps:0", "xpu:3", "tt:1", "musa:0", "npu:4")] == \
        [1, 2, 0, 3, 1, 0, 4]
    assert M.accelerator_index("mps") == 0 and M.accelerator_index("cuda:") == 0
    assert [M.accelerator_index(d) for d in ("cpu", "", None, "fgp:cuda:0,cuda:1", "meta")] == [None] * 5


def test_a_tensor_goes_where_its_name_says_and_a_compound_name_moves_nothing():
    if DeviceAllocator.device_count() == 0:          # only a machine without an accelerator skips; any other error fails
        pytest.skip("no accelerator on this machine")
    host = NBXTensor.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3))
    assert M.to_named_device(host, "fgp:cuda:0,cuda:1") is host
    on = M.to_named_device(host, "mps:0" if DeviceAllocator.device_count() == 1 else "cuda:0")
    assert on.is_cuda and on.device_idx == 0
    back = M.to_named_device(on, "cpu")
    assert back.is_cpu and back.numpy().tolist() == host.numpy().tolist()


def test_a_named_wait_selects_its_device_and_restores_the_current_one_even_when_it_raises(monkeypatch):
    current = {"dev": 0}
    seen = []
    monkeypatch.setattr(DeviceAllocator, "get_device", staticmethod(lambda: current["dev"]))
    monkeypatch.setattr(DeviceAllocator, "set_device", staticmethod(lambda d: current.update(dev=d)))
    monkeypatch.setattr(DeviceAllocator, "sync_device", staticmethod(lambda: seen.append(current["dev"])))
    M.synchronize_named("hip:3")
    assert seen == [3] and current["dev"] == 0
    M.synchronize_named("cpu")                                   # no accelerator named: the current device
    assert seen == [3, 0] and current["dev"] == 0

    def boom():
        raise RuntimeError("the device faulted")
    monkeypatch.setattr(DeviceAllocator, "sync_device", staticmethod(boom))
    with pytest.raises(RuntimeError, match="faulted"):
        M.synchronize_named("cuda:2")
    assert current["dev"] == 0
