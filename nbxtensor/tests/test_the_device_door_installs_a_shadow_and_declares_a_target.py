"""`nbxtensor.device` is an application's door to a shadow and to a declared compile target: `install_shadow(devices)`
turns the process into a shadow of that many devices (then `count()` and `current()` answer the shadow's), refused by
name while a device is in sight or for a count below one; `declare_target(maker, capability)` binds the compile target a
hardware description states (nvidia -> cuda, 32 threads a warp; amd -> hip, 64; "7.0" -> 70) and has the device
description read again, refused by name for any other maker or a capability that is not a number.

Before (2026-11-13): the engine's census reached into `memory.shadow` (`install_shadow`, `DevicesVisible`,
`shadow_device`) and into `backend.launcher.bind_target` with a `GPUTarget` it built itself.

What these tests do if the code is wrong: a door is missing, a shadow does not answer its count or current device, a
visible device or a count of zero is accepted, a target carries the wrong backend, capability or warp, the device
description is not read again, or a malformed capability is bound — red.

A shadow rewires the allocator for the whole process, so each shadow case runs in its own interpreter; the runtime's
device count is answered by a stand-in there (a host with no card, or one in sight), never asked of a driver.

Run: python -m pytest nbxtensor/tests/test_the_device_door_installs_a_shadow_and_declares_a_target.py
"""
import os
import subprocess
import sys
import textwrap

import pytest


def _in_own_interpreter(visible: int, body: str) -> subprocess.CompletedProcess:
    code = textwrap.dedent(f"""
        from nbxtensor.memory.allocator import DeviceAllocator as DA
        DA.device_count = staticmethod(lambda: {visible})     # a stand-in for the runtime's count
        from nbxtensor import device
    """) + textwrap.dedent(body)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", NBX_CENSUS="1")
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)


def test_a_shadow_answers_its_count_and_the_device_selected():
    r = _in_own_interpreter(0, """
        device.install_shadow(3)
        assert device.count() == 3 and device.current() == 0
        device.select(2)
        assert device.current() == 2
        device.install_shadow(5)                       # once per process: a second call changes nothing
        assert device.count() == 3
        print("SHADOW")
    """)
    assert r.returncode == 0 and "SHADOW" in r.stdout, r.stderr[-1500:]


def test_a_shadow_with_a_device_in_sight_is_refused_by_name():
    r = _in_own_interpreter(1, """
        try:
            device.install_shadow(1)
        except device.DevicesVisible as e:
            assert "1 device(s) visible" in str(e), e
            print("REFUSED")
    """)
    assert r.returncode == 0 and "REFUSED" in r.stdout, r.stderr[-1500:]


def test_a_shadow_of_no_device_is_refused_by_name():
    from nbxtensor import device
    with pytest.raises(ValueError, match=r"install_shadow\(0\): a shadow stands for at least one device"):
        device.install_shadow(0)


@pytest.fixture
def bound(monkeypatch):
    """The process's target and the description's re-reads, both restored after the test."""
    from nbxtensor.backend import device_files, launcher
    monkeypatch.setattr(launcher, "_TARGET", None)
    forgot = []
    monkeypatch.setattr(device_files, "forget", lambda: forgot.append(True))
    return launcher, forgot


@pytest.mark.parametrize("maker, capability, backend, arch, warp", [
    ("nvidia", "7.0", "cuda", 70, 32),
    ("NVIDIA", "8.6", "cuda", 86, 32),
    ("nvidia", "8", "cuda", 80, 32),
    ("amd", "9.0", "hip", 90, 64),
])
def test_a_declared_target_is_the_launchers_target(bound, maker, capability, backend, arch, warp):
    from nbxtensor import device
    launcher, forgot = bound
    device.declare_target(maker, capability)
    t = launcher.target()
    assert (t.backend, t.arch, t.warp_size) == (backend, arch, warp)
    assert forgot == [True]


@pytest.mark.parametrize("maker, capability", [
    ("apple", "0.0"), ("acme", "7.0"), ("nvidia", "sm70"), ("nvidia", "7.0.1"), ("nvidia", ""), ("amd", "7..0"),
])
def test_an_undeclarable_target_is_refused_by_name_and_binds_nothing(bound, maker, capability):
    from nbxtensor import device
    launcher, forgot = bound
    with pytest.raises(device.TargetNotDeclarable, match=r"declare_target\("):
        device.declare_target(maker, capability)
    assert launcher._TARGET is None and forgot == []
