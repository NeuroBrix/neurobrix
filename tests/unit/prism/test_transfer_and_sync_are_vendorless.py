"""Two guards whose failure was worse than "does not support Apple".

`ExecutionStrategy.transfer_tensor` matched only `"cuda:"` on the NBXTensor
path and sent everything else to the `else` branch, which returns
`tensor.to_cpu()`. Asked to move a tensor to `mps:0` or `hip:0`, it moved it
to the HOST — silently, and in the wrong direction. An unrecognised device now
raises instead of being read as a CPU request, because that fallback is what
made the bug invisible.

`TritonStrategy.synchronize_device` carried two defects, and only one of them
was about non-NVIDIA hardware:

  * a hip:1 / mps:0 device did not match, so no set_device was issued and the
    call synchronised whatever card was current — the wrong one;
  * set_device(idx) was NEVER RESTORED, so the call leaked a device change.
    That bites multi-GPU NVIDIA just as hard, and it does it by mutating
    global state.
"""

from __future__ import annotations

import pytest

from neurobrix.core.strategies.base import ExecutionStrategy


class _NBX:
    """Duck-types NBXTensor: to_cuda / to_cpu / _device."""

    def __init__(self):
        self._device = "cuda"
        self.moved = None

    def to_cuda(self, idx=0):
        self.moved = ("gpu", idx)
        return self

    def to_cpu(self):
        self.moved = ("cpu", None)
        return self


def _move(dev):
    t = _NBX()
    ExecutionStrategy.transfer_tensor(None, t, dev)
    return t.moved


@pytest.mark.parametrize("dev,idx", [("cuda:0", 0), ("cuda:3", 3), ("cuda", 0)])
def test_nvidia_transfer_is_unchanged(dev, idx):
    assert _move(dev) == ("gpu", idx)


@pytest.mark.parametrize("dev,idx", [("hip:0", 0), ("hip:2", 2), ("mps:0", 0), ("xpu:1", 1)])
def test_a_gpu_target_goes_to_the_gpu_not_the_host(dev, idx):
    assert _move(dev) == ("gpu", idx), (
        f"{dev} was moved to the host — the caller asked for a GPU"
    )


@pytest.mark.parametrize("dev", ["cpu", "cpu:0"])
def test_a_host_target_still_goes_to_the_host(dev):
    assert _move(dev) == ("cpu", None)


def test_an_unrecognised_device_raises_instead_of_going_to_the_host():
    with pytest.raises(ValueError, match="neither the host nor a known accelerator"):
        _move("wombat:0")


# --- synchronize_device -------------------------------------------------

class _DA:
    """Stand-in DeviceAllocator recording set_device / sync order."""

    current = 0
    calls: list = []

    @classmethod
    def get_device(cls):
        return cls.current

    @classmethod
    def set_device(cls, idx):
        cls.current = idx
        cls.calls.append(("set", idx))

    @classmethod
    def sync_device(cls):
        cls.calls.append(("sync", cls.current))


@pytest.fixture
def da(monkeypatch):
    import neurobrix.kernels.nbx_tensor as nbx
    _DA.current, _DA.calls = 0, []
    monkeypatch.setattr(nbx, "DeviceAllocator", _DA)
    return _DA


def _sync(dev):
    from neurobrix.core.strategies.triton.base import TritonStrategy
    TritonStrategy.synchronize_device(None, dev)


@pytest.mark.parametrize("dev,idx", [("cuda:1", 1), ("hip:2", 2), ("mps:0", 0), ("xpu:3", 3)])
def test_it_synchronises_the_card_it_was_asked_for(da, dev, idx):
    _sync(dev)
    assert ("sync", idx) in da.calls, (
        f"{dev}: synchronised {[c for c in da.calls if c[0]=='sync']} "
        f"instead of device {idx}"
    )


@pytest.mark.parametrize("dev", ["cuda:1", "hip:2", "mps:0"])
def test_it_restores_the_device_it_changed(da, dev):
    da.current = 0
    _sync(dev)
    assert da.current == 0, (
        f"{dev} left the current device at {da.current} — the next operation "
        f"that assumes device 0 runs on the wrong card"
    )


def test_no_device_named_means_sync_the_current_one_and_change_nothing(da):
    da.current = 2
    _sync(None)
    assert da.calls == [("sync", 2)]
    assert da.current == 2
