"""The access grant of an expandable range is built from the answers of the allocator's one peer door, and a question
that fails is refused by name — never read as "no peer".

A chunk mapped into a card's expandable range is readable only from the devices its grant names (`memory.vmm.Vmm`).
Before: a failing `cuDeviceGetCount` or peer question (status != 0) left the grant at the owning card alone, in
silence — the card-to-card copy of any tensor of the range then failed far from its cause ("cudaMemcpy failed rc=1
(kind=3)"). The peer question is asked through `DeviceAllocator.can_access_peer` alone (the allocator is the one door
to the device), and that door refuses a runtime error by name.

What this test would do if the code were wrong: a failing call builds a one-entry grant and raises nothing — RED on
the refusals; a device the door says is no peer refused, or a peer left out of the grant — RED on the two controls.
No card: the driver library and the runtime are stood in for."""
import ctypes

import pytest

import nbxtensor.backend.runtime as _rt
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.memory.vmm import Vmm


class _Driver:
    """A driver of three devices; `fail` maps a call to the status it answers."""

    def __init__(self, fail=None):
        self.fail = dict(fail or {})

    def cuDeviceGetCount(self, count):
        count._obj.value = 3
        return self.fail.get("cuDeviceGetCount", 0)


class _Runtime:
    """A runtime whose peer question answers `peers` for device 1's memory; `fail` maps an asking device to a status."""

    def __init__(self, peers, fail=None):
        self.peers, self.fail, self.asked = set(peers), dict(fail or {}), []

    def cudaDeviceCanAccessPeer(self, can, dev, peer):
        self.asked.append((dev.value, peer.value))
        can._obj.value = 1 if dev.value in self.peers else 0
        return self.fail.get(dev.value, 0)

    def cudaGetLastError(self):
        return 0


@pytest.fixture
def runtime(monkeypatch):
    def install(peers, fail=None):
        rt = _Runtime(peers, fail)
        monkeypatch.setattr(_rt, "gpu_runtime", lambda: rt)
        monkeypatch.setattr(_rt, "backend_table", lambda: _rt._GPU_BACKENDS["cuda"])
        monkeypatch.setattr(DeviceAllocator, "clear_last_error", staticmethod(lambda: 0))
        return rt
    return install


def _granted(vmm):
    return [(a.location.id, a.flags) for a in vmm._access]


def test_every_peer_the_door_names_is_granted(runtime):
    rt = runtime(peers={0, 2})
    assert _granted(Vmm(_Driver(), 1)) == [(1, 3), (0, 3), (2, 3)]
    assert rt.asked == [(0, 1), (2, 1)]          # each other card asked whether it reaches card 1's memory


def test_a_device_the_door_says_is_no_peer_is_left_out_without_a_refusal(runtime):
    runtime(peers={2})
    assert _granted(Vmm(_Driver(), 1)) == [(1, 3), (2, 3)]


def test_a_failed_device_count_is_refused_by_name(runtime):
    runtime(peers={0, 2})
    with pytest.raises(RuntimeError, match=r"cuDeviceGetCount \(status 3\).*device 1's expandable range"):
        Vmm(_Driver(fail={"cuDeviceGetCount": 3}), 1)


@pytest.mark.parametrize("asking", [0, 2])
def test_a_failed_peer_question_is_refused_by_name(runtime, asking):
    runtime(peers={0, 2}, fail={asking: 999})
    with pytest.raises(RuntimeError,
                       match=rf"cudaDeviceCanAccessPeer for device {asking} -> 1 \(error 999\).*device 1's expandable"):
        Vmm(_Driver(), 1)


def test_the_stand_in_speaks_ctypes_as_the_driver_does():
    count = ctypes.c_int(0)
    assert _Driver().cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value == 3
