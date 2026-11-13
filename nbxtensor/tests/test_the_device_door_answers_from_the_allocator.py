"""`nbxtensor.device` is the application's door to the devices: each name answers from the allocator that holds the
mechanism, with the same arguments, and a refusal of the runtime reaches the caller unchanged.

Before (2026-11-13): the engine called `DeviceAllocator.set_device`, `get_device`, `ensure_triton_device`,
`sync_device`, `device_synchronize`, `mem_get_info`, `device_count` and `most_free_device` itself, 64 times in 25
files: the library's allocator class used as the engine's device API (the red line: the engine keeps policy and asks
the library through its public names).

What this test does if the code is wrong: a name calls another method, drops or changes its argument, or swallows a
refusal — red.

Run: python -m pytest nbxtensor/tests/test_the_device_door_answers_from_the_allocator.py
"""
import pytest


@pytest.fixture
def calls(monkeypatch):
    from nbxtensor.memory.allocator import DeviceAllocator
    seen = []

    def record(name, answer=None):
        def method(*args):
            seen.append((name, args))
            return answer
        monkeypatch.setattr(DeviceAllocator, name, staticmethod(method))

    record("device_count", 3)
    record("get_device", 2)
    record("set_device")
    record("ensure_triton_device")
    record("sync_device")
    record("device_synchronize")
    record("mem_get_info", (5, 9))
    record("most_free_device", 1)
    return seen


def test_each_name_answers_from_its_mechanism(calls):
    from nbxtensor import device
    assert device.count() == 3 and device.current() == 2 and device.most_free() == 1
    assert device.mem_info(1) == (5, 9) and device.mem_info() == (5, 9)
    device.select(1)
    device.select_for_launch(1)
    device.synchronize()
    device.drain(0)
    device.drain()
    assert calls == [("device_count", ()), ("get_device", ()), ("most_free_device", ()), ("mem_get_info", (1,)),
                     ("mem_get_info", (None,)), ("set_device", (1,)), ("ensure_triton_device", (1,)),
                     ("sync_device", ()), ("device_synchronize", (0,)), ("device_synchronize", (None,))]


def test_a_refusal_reaches_the_caller_unchanged(monkeypatch):
    from nbxtensor import device
    from nbxtensor.memory.allocator import DeviceAllocator

    def refuse(index):
        raise RuntimeError(f"set_device({index}) refused by the runtime (error 101)")
    monkeypatch.setattr(DeviceAllocator, "set_device", staticmethod(refuse))
    with pytest.raises(RuntimeError, match=r"set_device\(7\) refused by the runtime \(error 101\)"):
        device.select(7)


def test_the_door_exports_exactly_its_names():
    from nbxtensor import device
    assert sorted(device.__all__) == sorted(["count", "current", "select", "select_for_launch", "synchronize",
                                             "drain", "mem_info", "most_free", "target_name", "DeviceQueryError",
                                             "NoGpuRuntime"])
    assert all(callable(getattr(device, n)) for n in device.__all__)
