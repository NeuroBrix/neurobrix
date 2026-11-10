"""`DeviceAllocator.peer_copy`: the device-to-device copy on raw pointers a caller whose tensors are not this library's
needs (the compiled engine: PyTorch >= 2.10 creates a context on EVERY visible card at its first GPU-to-GPU copy —
306 MiB on cards a plan does not name, measured 2026-11-10). Its contract is an ORDER and a set of devices:
the source synchronised before the copy (the copy does not wait for it), the copy issued under the destination, the
destination synchronised before returning (the caller frees the source at once), no device selected but those two,
the device found current never "restored" (that is how a context lands on card 0).
What this would do if the code were wrong: a stray context on a third card, a copy of a source still being written,
or a source freed under a copy still queued."""
import numpy as np
import pytest

from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.tensor import NBXTensor


@pytest.fixture
def calls(monkeypatch):
    log = []
    monkeypatch.setattr(DeviceAllocator, "set_device", staticmethod(lambda d: log.append(("select", int(d)))))
    monkeypatch.setattr(DeviceAllocator, "sync_device", staticmethod(lambda: log.append(("sync",))))
    monkeypatch.setattr(DeviceAllocator, "ensure_peer_access",
                        staticmethod(lambda s, d: (log.append(("grant", int(s), int(d))), True)[1]))
    monkeypatch.setattr(DeviceAllocator, "memcpy",
                        staticmethod(lambda dst, src, n, kind=3: log.append(("copy", dst, src, n, kind))))
    return log


def test_the_source_is_synchronised_first_the_copy_runs_under_the_destination_which_is_synchronised_last(calls):
    assert DeviceAllocator.peer_copy(0xD000, 3, 0x5000, 2, 4096) is True
    assert calls == [("select", 2), ("sync",), ("grant", 2, 3), ("select", 3), ("copy", 0xD000, 0x5000, 4096, 3),
                     ("sync",)]


def test_no_device_but_the_two_is_ever_selected_and_the_destination_stays_selected(calls):
    DeviceAllocator.peer_copy(0xD000, 3, 0x5000, 2, 64)
    selected = [c[1] for c in calls if c[0] == "select"]
    assert set(selected) == {2, 3} and selected[-1] == 3


def test_a_refused_grant_is_an_answer_not_an_error_and_the_copy_still_runs(calls, monkeypatch):
    monkeypatch.setattr(DeviceAllocator, "ensure_peer_access", staticmethod(lambda s, d: False))
    assert DeviceAllocator.peer_copy(0xD000, 3, 0x5000, 2, 64) is False
    assert ("copy", 0xD000, 0x5000, 64, 3) in calls


def test_nothing_to_copy_copies_nothing_and_a_null_pointer_is_refused_by_name(calls):
    DeviceAllocator.peer_copy(0, 3, 0, 2, 0)
    assert not [c for c in calls if c[0] == "copy"]
    with pytest.raises(ValueError, match="peer_copy"):
        DeviceAllocator.peer_copy(0, 3, 0x5000, 2, 16)


def test_the_bytes_arrive():
    """On the device at hand (one device: the copy is the same call, the grant is trivially true)."""
    src = NBXTensor.from_numpy(np.arange(4096, dtype=np.float32).reshape(64, 64))
    dst = NBXTensor.from_numpy(np.zeros((64, 64), np.float32))
    dev = int(src._device_idx)
    assert DeviceAllocator.peer_copy(dst.data_ptr(), dev, src.data_ptr(), dev, src._nbytes) is True
    assert np.array_equal(dst.numpy(), src.numpy())
