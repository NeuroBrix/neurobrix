"""Whether this host has no device to give a test, and the one conversion of a failure into a skip that says so.

THE ONE PLACE the suites learn it: `nbxtensor/tests/conftest.py` and the engine's `tests/conftest.py` both register the
hooks below. A card test does not ask "is a card visible" its own way: on a host whose GPU runtime library is installed
without its driver, the runtime refuses to count its devices (CUDA error 35) and the library raises `DeviceQueryError`
by name. A test that asked only "how many cards" failed there instead of skipping: 169 cells of the CPU suite
(2026-11-13), red for the absence of a card and not for a result.

HOW "NO DRIVER" IS TOLD FROM "A BROKEN DRIVER"
-----------------------------------------------
The question is the library's own device door, `nbxtensor.device.count()`, executed once per session:

- it answers N > 0: a host with cards. Nothing is armed; every failure stays a failure.
- it answers 0 (no GPU runtime, or the runtime's own "no device" code, error 100 under `CUDA_VISIBLE_DEVICES=`): no
  device visible to this process. With no GPU runtime at all (a CPU install without the runtime wheel) the backend
  table cannot be read either (`NoGpuRuntime`): the skip says so ("needs a GPU device: no GPU runtime on this host")
  and the runtime's own phrase ("No GPU runtime found") is the mark. Nothing here raises in `pytest_configure`.
- it raises `DeviceQueryError`: the runtime is there and refuses. Then the driver libraries the backend table names
  (`backend_table()["driver_libs"]`) are loaded: if NONE loads, there is no driver on this host, and the skip names
  the backend and the runtime's code ("needs a CUDA device: none answered (driver error 35)"). If one loads, a driver
  is installed and refuses: a broken driver, never a skip: nothing is armed. A table that names no driver library
  cannot tell the two apart and arms nothing either.

WHAT IS CONVERTED
-----------------
Only on an armed host, and only a failure whose own text says the device was refused: the runtime's code that the
door answered ("error 35"), the "no device" code ("error 100"), or a vendor's phrase for the same fact (torch says
"Found no NVIDIA driver"). Any other failure on that host fails as it does anywhere. A test that FAILS at setup (a
fixture that allocates) or at import (a module that asks for a device at its top) is converted the same way. A test
marked `cardless` is about a host without a card (the census shadow): its refusal is its result and is never
converted.
"""
from __future__ import annotations

import ctypes
import functools
import re
from typing import Optional

import pytest

CARDLESS = ("cardless: the test's subject is a host without a card (the census shadow); a refusal of the device door "
            "there is its failure, never a skip")

# Each vendor phrases the absence its own way; torch's reaches the suite through the bridge tests.
_VENDOR_MARKS = ("cudaErrorNoDevice", "driver_total=0MB", "No GPU runtime found", "No CUDA GPUs are available",
                 "Found no NVIDIA driver")


@functools.lru_cache(maxsize=1)
def _state() -> tuple:
    """(reason, marks): the skip reason and the texts that name the refusal, or (None, ()) when nothing is armed."""
    from nbxtensor import device
    from nbxtensor.backend import runtime
    try:
        n = device.count()
    except device.DeviceQueryError as exc:
        libs = runtime.backend_table().get("driver_libs") or []
        if not libs or any(_loads(lib) for lib in libs):
            return None, ()                 # a driver answers ctypes and refuses the runtime: broken, a failure
        code = re.search(r"error (\d+)", str(exc))
        if code is None:
            return None, ()                 # a refusal with no runtime code: nothing to recognise it by
        rc = code.group(1)
        return (f"needs a {runtime.gpu_backend().upper()} device: none answered (driver error {rc})",
                (re.compile(rf"\berror {rc}\b"),))
    if n > 0:
        return None, ()
    try:
        no_device = runtime.backend_table().get("no_device_rc")
    except runtime.NoGpuRuntime as exc:
        # no runtime at all (a CPU install with no runtime wheel): the same class as "no driver", named for what it is
        return f"needs a GPU device: no GPU runtime on this host ({str(exc).splitlines()[0]})", ()
    marks = (re.compile(rf"\berror {no_device}\b"),) if no_device is not None else ()
    return "no GPU visible to this process", marks


def _loads(lib: str) -> bool:
    try:
        ctypes.CDLL(lib)
    except OSError:
        return False
    return True


def absence() -> Optional[str]:
    """The reason this host gives no device to a test, or None on a host with one (or a broken driver)."""
    return _state()[0]


def _names_the_refusal(text: str) -> bool:
    _, marks = _state()
    return any(m.search(text) for m in marks) or any(m in text for m in _VENDOR_MARKS)


def _skip_if_refused(item, exc: BaseException) -> None:
    reason = absence()
    if reason is None or item.get_closest_marker("cardless") is not None:
        return
    text = f"{type(exc).__name__}: {exc}"
    if _names_the_refusal(text):
        # reported at the test's own line, not this one: a summary folded on one helper line would hide which tests
        raise pytest.skip.Exception(f"{reason}: {text[:200]}", _use_item_location=True)


def pytest_configure(config):
    config.addinivalue_line("markers", CARDLESS)
    _state()        # asked once, before any test can monkeypatch the door it asks


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    try:
        return (yield)
    except Exception as exc:                              # noqa: BLE001 — re-raised unless it names the refusal
        _skip_if_refused(item, exc)
        raise


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except Exception as exc:                              # noqa: BLE001 — re-raised unless it names the refusal
        _skip_if_refused(item, exc)
        raise


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    """A module that asks for a device at import: its collection error becomes the same skip, by name."""
    report = yield
    if not report.failed or not isinstance(collector, pytest.Module):
        return report
    reason = absence()
    text = str(report.longrepr)
    if reason is None or not _names_the_refusal(text):
        return report
    return pytest.CollectReport(report.nodeid, "skipped", (str(collector.path), 0, f"Skipped: {reason}"), [])
