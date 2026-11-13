"""The name a process compiles for (`nbxtensor.device.target_name`) and the pool's counters
(`nbxtensor.memory.pool_stats`) are public doors: an application (the kernel certifier) reads them there, never
through `backend.launcher` or `DeviceAllocator`.

Before (2026-11-13): the certifier built `<backend>-<arch>` from `backend.launcher.target` and `arch_name` itself and
read `DeviceAllocator.pool_stats` directly (the red line: the engine asks the library through its public names).

What this test does if the code is wrong: a door missing, a name built from another field, a counter set handed out
as the pool's own dict, or a driver's refusal swallowed — red.

Run: python -m pytest nbxtensor/tests/test_the_compile_target_and_the_pool_s_counters_are_read_through_the_doors.py
"""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("backend, arch, name", [("cuda", 70, "cuda-70"), ("hip", "gfx90a", "hip-gfx90a"),
                                                  ("metal", "apple-m4-pro", "metal-apple-m4-pro"),
                                                  ("xpu", {"arch": "pvc", "max_work_group_size": 1024}, "xpu-pvc")])
def test_the_target_name_is_the_backend_and_the_arch(monkeypatch, backend, arch, name):
    from nbxtensor import device
    from nbxtensor.backend import launcher
    monkeypatch.setattr(launcher, "target", lambda: SimpleNamespace(backend=backend, arch=arch, warp_size=32))
    assert device.target_name() == name


def test_a_driver_that_cannot_answer_reaches_the_caller(monkeypatch):
    from nbxtensor import device
    from nbxtensor.backend import launcher

    def refuse():
        raise RuntimeError("cuDeviceGetAttribute refused (error 3)")
    monkeypatch.setattr(launcher, "target", refuse)
    with pytest.raises(RuntimeError, match=r"cuDeviceGetAttribute refused \(error 3\)"):
        device.target_name()


def test_the_pool_s_counters_are_a_copy_of_the_allocator_s(monkeypatch):
    from nbxtensor import memory
    from nbxtensor.memory.allocator import DeviceAllocator
    held = {"exact": 3, "fit": 1, "miss": 2, "flushes": 1, "evictions": 0}
    monkeypatch.setattr(DeviceAllocator, "_POOL", SimpleNamespace(stats=held))
    got = memory.pool_stats()
    assert got == held and got is not held
    got["flushes"] = 99
    assert held["flushes"] == 1
