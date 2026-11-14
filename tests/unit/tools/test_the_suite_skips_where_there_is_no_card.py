"""The suite-wide hook turns "there is no card" into a skip, and turns nothing else.

Measured 2026-09-16: the unit suite under the workshop's device door
(`CUDA_VISIBLE_DEVICES=`) returned 50 failures, **42 of them one thing** — a test
allocating on a device that is not there, raising `DeviceOOMError: GPU malloc failed
(error 100)`, error 100 being `cudaErrorNoDevice`. A suite that answers fifty reds
meaning "no card here" is a suite nobody runs beside a campaign, which is the only
place the door exists for.

`tests/conftest.py` converts that ONE failure into a skip, through the hooks of
`nbxtensor/tests/_no_device.py` (the library suite registers the same ones). This file is what
keeps the conversion from becoming a swallow, and it pins both directions:

* a failure whose text carries a no-device mark becomes a skip — **only** when an
  executing probe says this process can see no device;
* every other failure stays a failure, at the same moment, in the same run;
* on a machine that HAS a device the hook is disarmed entirely, so a real allocation
  failure on a real card is untouched;
* on a host whose GPU runtime is installed WITHOUT its driver (the door raises
  `DeviceQueryError`, CUDA error 35, and no driver library of the backend table loads) the
  skip names the backend and the code; where a driver library loads and the door still
  refuses, the driver is broken and nothing is armed.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
NO_DEVICE = ROOT / "nbxtensor" / "tests" / "_no_device.py"

_TWO_TESTS = '''
def test_it_cannot_reach_a_card():
    raise RuntimeError("GPU malloc failed (error 100) for 64 bytes [driver_total=0MB]")

def test_it_is_simply_wrong():
    assert 2 + 2 == 5, "this is arithmetic, not a missing card"
'''


def _run(tmp_path: Path, visible: str | None):
    (tmp_path / "_no_device.py").write_text(NO_DEVICE.read_text())
    (tmp_path / "conftest.py").write_text(
        "from _no_device import pytest_configure, pytest_make_collect_report, pytest_runtest_call, "
        "pytest_runtest_setup\n")
    (tmp_path / "test_two.py").write_text(_TWO_TESTS)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT / "nbxtensor" / "src")))
    if visible is None:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = visible
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
                          capture_output=True, text=True, env=env, timeout=300, cwd=str(tmp_path))


def _devices_visible_under(visible: str) -> int:
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = visible; env["PYTHONPATH"] = str(ROOT / "src")
    have = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r);"
         "import neurobrix; from nbxtensor.memory.allocator import DeviceAllocator; print(DeviceAllocator.device_count())"
         % str(ROOT / "src")],
        capture_output=True, text=True, timeout=180, env=env)
    return int(have.stdout.strip() or 0) if have.returncode == 0 else 0


def test_with_no_card_visible_only_the_no_card_failure_becomes_a_skip(tmp_path):
    if _devices_visible_under("") > 0:
        # The door is CUDA's: an empty CUDA_VISIBLE_DEVICES hides nothing from a Metal host, the
        # hook stays disarmed (correctly — the other test below shows it) and both inner tests
        # fail. The closed door cannot be shown from here (the Mac, 2026-09-28: "2 failed").
        pytest.skip("a device stays visible under CUDA_VISIBLE_DEVICES='' on this host (not a CUDA "
                    "backend): the no-card door cannot be closed from here")
    out = _run(tmp_path, "")
    tail = out.stdout.strip().splitlines()[-1]
    assert "1 failed" in tail and "1 skipped" in tail, (
        f"expected exactly one skip (the missing card) and one failure (the arithmetic); got: {tail}")
    assert "this is arithmetic" in out.stdout, "the real failure must still be reported in full"


def test_with_a_card_visible_the_hook_is_disarmed(tmp_path):
    """The half that keeps it from being a swallow: on a real machine, nothing is converted."""
    env2 = dict(os.environ); env2.pop("CUDA_VISIBLE_DEVICES", None); env2["PYTHONPATH"] = str(ROOT / "src")
    have = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r);"
         "import neurobrix; from nbxtensor.memory.allocator import DeviceAllocator; print(DeviceAllocator.device_count())"
         % str(ROOT / "src")],
        capture_output=True, text=True, timeout=180, env=env2)
    if have.returncode != 0 or int(have.stdout.strip() or 0) == 0:
        pytest.skip("no GPU visible to this host — the disarmed half cannot be shown from here")
    out = _run(tmp_path, None)
    tail = out.stdout.strip().splitlines()[-1]
    assert "2 failed" in tail, (
        f"with a device visible BOTH failures must stand — the hook must not convert "
        f"anything on a real machine; got: {tail}")


def _the_door(monkeypatch, *, answers=None, refuses=None, driver_libs=()):
    """The helper's question asked against a door that answers `answers` or refuses with `refuses`, and a backend
    table naming `driver_libs`; the session's cached answer is forgotten before and after."""
    import importlib.util
    from nbxtensor import device
    from nbxtensor.backend import runtime
    spec = importlib.util.spec_from_file_location("_no_device_under_test", NO_DEVICE)
    nd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nd)

    def count():
        if refuses is not None:
            raise device.DeviceQueryError(refuses)
        return answers
    monkeypatch.setattr(device, "count", count)
    monkeypatch.setattr(runtime, "gpu_backend", lambda: "cuda")
    monkeypatch.setattr(runtime, "backend_table", lambda: {"driver_libs": list(driver_libs), "no_device_rc": 100})
    return nd


_REFUSED = ("cudaGetDeviceCount failed with error 35 on cuda: the runtime is there and refuses to count its devices "
            "(a driver older than the runtime, or a broken install)")


def test_a_runtime_without_its_driver_is_a_skip_that_names_the_backend_and_the_code(monkeypatch):
    nd = _the_door(monkeypatch, refuses=_REFUSED, driver_libs=["libnbx-no-such-driver.so.1"])
    assert nd.absence() == "needs a CUDA device: none answered (driver error 35)"
    assert nd._names_the_refusal("RuntimeError: set_device(0) refused by the runtime (error 35)")
    assert not nd._names_the_refusal("RuntimeError: cudaMalloc failed with error 350")
    assert not nd._names_the_refusal("AssertionError: this is arithmetic, not a missing card")


def test_a_driver_that_loads_and_refuses_is_broken_and_arms_nothing(monkeypatch):
    import ctypes.util
    present = ctypes.util.find_library("c") or ctypes.util.find_library("System")
    assert present, "a library every host loads stands for an installed driver"
    nd = _the_door(monkeypatch, refuses=_REFUSED, driver_libs=["libnbx-no-such-driver.so.1", present])
    assert nd.absence() is None and not nd._names_the_refusal("error 35")


def test_a_table_with_no_driver_library_cannot_tell_and_arms_nothing(monkeypatch):
    nd = _the_door(monkeypatch, refuses=_REFUSED, driver_libs=())
    assert nd.absence() is None


def test_a_door_that_counts_a_card_arms_nothing_and_one_that_counts_none_says_so(monkeypatch):
    assert _the_door(monkeypatch, answers=2).absence() is None
    nd = _the_door(monkeypatch, answers=0)
    assert nd.absence() == "no GPU visible to this process" and nd._names_the_refusal("malloc failed (error 100)")
