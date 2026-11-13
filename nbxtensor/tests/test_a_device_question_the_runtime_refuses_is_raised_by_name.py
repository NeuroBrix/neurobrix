"""A device question the runtime refuses is raised by name (`DeviceQueryError`, the call, the device, the runtime's
code); only the absence of a GPU runtime, or the runtime's own "no device" code, reads as no device.

Before (2026-11-13, found moving `nbxtensor.device`): `DeviceAllocator.device_count` answered 0 for ANY failure, so a
driver older than its runtime read as "no GPU"; `most_free_device` and `visible_device_memory` skipped a device whose
query failed (the walk then picked among the others, or listed fewer devices), and a failed restore of the current
device was not raised.

qa (2026-11-13), on the same family: `free_memory_mb` and `device_total_bytes` ignored the runtime's code (0.0 and 0
for rc=700) and swallowed the refusal (None, -1), and Prism's budget reader then built a card of 0 MB from a broken
driver; `mem_get_info` raised a plain RuntimeError; an interrupt in the walk skipped the restore.

What this test does if the code is wrong: a refusal comes back as 0, None, -1, a smaller list or another device, a
card of 0 MB reaches Prism, or the current device is not restored after a failed or interrupted read — red.

Run: python -m pytest nbxtensor/tests/test_a_device_question_the_runtime_refuses_is_raised_by_name.py
"""
import pytest

from nbxtensor.backend import runtime as R
from nbxtensor.memory import allocator as A
from nbxtensor.memory.allocator import DeviceAllocator, DeviceQueryError

TABLE = {"device_count": "cudaGetDeviceCount", "set_device": "cudaSetDevice", "mem_get_info": "cudaMemGetInfo",
         "no_device_rc": 100}


class FakeRuntime:
    """The memory primitives of a runtime with `free` bytes free per device; `refuse` maps a call (and a device) to
    the runtime's code."""

    def __init__(self, free, refuse=None):
        self.free, self.refuse, self.current = list(free), dict(refuse or {}), 0

    def device_count(self):
        return self.refuse.get("count", 0), len(self.free)

    def get_device(self):
        return 0, self.current

    def set_device(self, i):
        rc = self.refuse.get(("select", i), 0)
        if rc == 0:
            self.current = i
        return rc

    def mem_get_info(self):
        return self.refuse.get(("info", self.current), 0), self.free[self.current], 32 << 30


@pytest.fixture
def runtime(monkeypatch):
    restored = []

    def install(free, refuse=None, restore_fails=False):
        fake = FakeRuntime(free, refuse)
        monkeypatch.setattr(R, "device_memory", lambda: fake)
        monkeypatch.setattr(R, "backend_table", lambda: TABLE)
        monkeypatch.setattr(R, "gpu_backend", lambda: "cuda")
        monkeypatch.setattr(R, "invalidate_current_device_cache", lambda: None)
        monkeypatch.setattr(DeviceAllocator, "clear_last_error", staticmethod(lambda: None))

        def restore(prev):
            if restore_fails:
                raise RuntimeError("cudaSetDevice(0) failed with error 46")
            restored.append(prev)
            fake.current = prev
        monkeypatch.setattr(DeviceAllocator, "restore_device", staticmethod(restore))
        return fake
    install.restored = restored
    return install


def test_no_gpu_runtime_is_no_device(monkeypatch):
    def absent():
        raise R.NoGpuRuntime("No GPU runtime found")
    monkeypatch.setattr(R, "device_memory", absent)
    assert DeviceAllocator.device_count() == 0
    assert DeviceAllocator.visible_device_memory() == []


def test_the_runtimes_own_no_device_code_is_no_device(runtime):
    runtime([1, 2], refuse={"count": 100})
    assert DeviceAllocator.device_count() == 0


def test_any_other_refusal_to_count_is_raised_with_its_code(runtime):
    runtime([1, 2], refuse={"count": 35})
    with pytest.raises(DeviceQueryError, match=r"cudaGetDeviceCount failed with error 35 on cuda"):
        DeviceAllocator.device_count()


def test_the_most_free_device_is_chosen_the_lowest_on_a_tie(runtime):
    fake = runtime([5, 9, 9, 2])
    assert DeviceAllocator.most_free_device() == 1
    assert set(runtime.restored) == {0} and fake.current == 0


def test_a_device_that_cannot_be_selected_is_named_never_skipped(runtime):
    fake = runtime([5, 9, 7], refuse={("select", 1): 101})
    with pytest.raises(DeviceQueryError, match=r"cudaSetDevice\(1\) failed with error 101"):
        DeviceAllocator.most_free_device()
    assert runtime.restored == [0] and fake.current == 0          # back where it was, the walk failed or not


def test_a_device_whose_memory_cannot_be_read_is_named_never_skipped(runtime):
    fake = runtime([5, 9, 7], refuse={("info", 2): 2})
    with pytest.raises(DeviceQueryError, match=r"cudaMemGetInfo failed with error 2 on device 2"):
        DeviceAllocator.visible_device_memory()
    assert set(runtime.restored) == {0} and fake.current == 0


def test_a_failed_restore_is_raised(runtime):
    runtime([5, 9], restore_fails=True)
    with pytest.raises(DeviceQueryError, match=r"the current device 0 could not be restored after reading device 1"):
        DeviceAllocator.most_free_device()


def test_a_failed_restore_after_a_failed_walk_is_said_on_the_walks_error(runtime):
    runtime([5, 9], refuse={("info", 1): 2}, restore_fails=True)
    with pytest.raises(DeviceQueryError, match=r"on device 1; and the current device 0 could not be restored after it: "
                                               r"cudaSetDevice\(0\) failed with error 46"):
        DeviceAllocator.most_free_device()


def test_the_devices_and_their_totals_are_listed(runtime):
    runtime([5, 9])
    assert DeviceAllocator.visible_device_memory() == [(0, 32 << 30), (1, 32 << 30)]


def test_a_refusal_beside_a_refused_selection_is_said_not_raised_instead(runtime, monkeypatch):
    fake = runtime([5], refuse={("select", 4): 101, "count": 35})
    monkeypatch.setattr(A, "_rt", R)
    with pytest.raises(RuntimeError, match=r"set_device\(4\) refused by the runtime \(error 101\) — and counting"):
        DeviceAllocator.set_device(4)
    assert fake.current == 0


BROKEN = {("info", 0): 700, ("info", 1): 700}          # cudaErrorIllegalAddress: a driver that refuses every read


@pytest.mark.parametrize("read", [lambda: DeviceAllocator.free_memory_mb(1), lambda: DeviceAllocator.device_total_bytes(1),
                                  lambda: DeviceAllocator.device_free_bytes(1), lambda: DeviceAllocator.mem_get_info(1),
                                  lambda: DeviceAllocator.device_total_bytes(), lambda: DeviceAllocator.device_free_bytes()],
                         ids=["free_memory_mb", "device_total_bytes", "device_free_bytes", "mem_get_info",
                              "total of the current", "free of the current"])
def test_a_runtime_answering_700_is_raised_never_read_as_zero(runtime, read):
    fake = runtime([5, 9], refuse=BROKEN)
    with pytest.raises(DeviceQueryError, match=r"cudaMemGetInfo failed with error 700 on device"):
        read()
    assert fake.current == 0


def test_a_device_the_process_does_not_see_is_refused_never_none(runtime):
    runtime([5, 9])
    with pytest.raises(DeviceQueryError, match=r"device 7: the process sees 2 device"):
        DeviceAllocator.free_memory_mb(7)


def test_prism_never_builds_a_card_of_0_mb_from_a_broken_driver(runtime, monkeypatch):
    runtime([5, 9], refuse=BROKEN)
    from neurobrix.core.prism import memory_budget as B
    with pytest.raises(DeviceQueryError, match=r"error 700"):
        B._read_through_the_runtime(1, None)


def test_an_interrupt_during_a_read_still_restores_the_current_device(runtime):
    fake = runtime([5, 9])
    read = fake.mem_get_info

    def interrupted():                           # the read of device 1, after the walk switched to it
        if fake.current == 1:
            raise KeyboardInterrupt
        return read()
    fake.mem_get_info = interrupted
    with pytest.raises(KeyboardInterrupt):
        DeviceAllocator.most_free_device()
    assert runtime.restored == [0] and fake.current == 0
