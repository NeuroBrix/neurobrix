"""The access grant of an expandable range is built from the driver's answers, and a driver call that fails is refused
by name — never read as "no peer".

A chunk mapped into a card's expandable range is readable only from the devices its grant names (`memory.vmm.Vmm`).
Before: a failing `cuDeviceGetCount`, `cuDeviceGet` or `cuDeviceCanAccessPeer` (status != 0) left the grant at the
owning card alone, in silence — the card-to-card copy of any tensor of the range then failed far from its cause
("cudaMemcpy failed rc=1 (kind=3)").

What this test would do if the code were wrong: each failing call builds a one-entry grant and raises nothing — RED on
the three refusals; a driver that answers "not a peer" with status 0 refused, or a peer left out of the grant — RED on
the two controls. No card: the driver library is stood in for."""
import ctypes

import pytest

from nbxtensor.memory.vmm import Vmm


class _Driver:
    """A driver of three devices; `fail` maps a call (or (call, device)) to the status it answers."""

    def __init__(self, peers, fail=None):
        self.peers, self.fail = set(peers), dict(fail or {})

    def cuDeviceGetCount(self, count):
        count._obj.value = 3
        return self.fail.get("cuDeviceGetCount", 0)

    def cuDeviceGet(self, dev, ordinal):
        dev._obj.value = 100 + ordinal
        return self.fail.get(("cuDeviceGet", ordinal), 0)

    def cuDeviceCanAccessPeer(self, can, dev, owner):
        other = dev.value - 100
        can._obj.value = 1 if other in self.peers else 0
        return self.fail.get(("cuDeviceCanAccessPeer", other), 0)


def _granted(vmm):
    return [(a.location.id, a.flags) for a in vmm._access]


def test_every_peer_the_driver_names_is_granted():
    assert _granted(Vmm(_Driver(peers={0, 2}), 1)) == [(1, 3), (0, 3), (2, 3)]


def test_a_device_the_driver_says_is_no_peer_is_left_out_without_a_refusal():
    assert _granted(Vmm(_Driver(peers={2}), 1)) == [(1, 3), (2, 3)]


@pytest.mark.parametrize("fail, named", [
    ({"cuDeviceGetCount": 3}, r"cuDeviceGetCount \(status 3\)"),
    ({("cuDeviceGet", 1): 101}, r"cuDeviceGet for device 1 \(status 101\)"),
    ({("cuDeviceGet", 2): 101}, r"cuDeviceGet for device 2 \(status 101\)"),
    ({("cuDeviceCanAccessPeer", 0): 999}, r"cuDeviceCanAccessPeer for device 0 \(status 999\)"),
])
def test_a_failed_driver_call_is_refused_by_name(fail, named):
    with pytest.raises(RuntimeError, match=named):
        Vmm(_Driver(peers={0, 2}, fail=fail), 1)


def test_the_stand_in_speaks_ctypes_as_the_driver_does():
    count = ctypes.c_int(0)
    assert _Driver(peers=()).cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value == 3
