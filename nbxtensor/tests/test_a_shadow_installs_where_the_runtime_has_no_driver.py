"""`nbxtensor.device.install_shadow` installs on a host whose GPU runtime is there without its driver: the runtime
refuses to count its devices (CUDA error 35, `DeviceQueryError`) and no driver library the backend table names loads,
so no device is in sight, which is the shadow's own case. Where a driver library loads and the runtime still refuses,
the driver is broken: the `DeviceQueryError` stands, by name. A backend table that names no driver library cannot tell
the two apart: the refusal stands.

Before (2026-11-14): `install_shadow` let the count's `DeviceQueryError` through on every such host, so the census
shadow, built to run with no card, refused on a CPU host that has the runtime wheel and no driver.

What these tests do if the code is wrong: the driverless shadow raises `DeviceQueryError` instead of answering its count,
or a refusing driver that loads (or a table with no driver library) is read as "no device" and the shadow installs —
red.

Each case runs in its own interpreter (a shadow rewires the allocator for the whole process); the runtime's refusal and
the driver libraries are stand-ins there, never asked of a driver.

Run: python -m pytest nbxtensor/tests/test_a_shadow_installs_where_the_runtime_has_no_driver.py
"""
import ctypes.util
import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.cardless

_LOADS = ctypes.util.find_library("c") or "libc.so.6"     # a library every host loads: stands for an installed driver


def _refusing_runtime(driver_libs, body: str) -> subprocess.CompletedProcess:
    code = textwrap.dedent(f"""
        from nbxtensor.memory.allocator import DeviceAllocator as DA, DeviceQueryError
        from nbxtensor.backend import runtime as rt
        def _refuses():
            raise DeviceQueryError("cudaGetDeviceCount failed with error 35 on cuda: the runtime is there and refuses")
        DA.device_count = staticmethod(_refuses)              # a stand-in for the runtime's refusal
        _table = rt.backend_table
        rt.backend_table = lambda: dict(_table(), driver_libs={driver_libs!r})
        from nbxtensor import device
    """) + textwrap.dedent(body)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", NBX_CENSUS="1")
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)


def test_a_runtime_without_its_driver_takes_a_shadow():
    r = _refusing_runtime(["libnbx-no-such-driver.so.1"], """
        device.install_shadow(2)
        assert device.count() == 2, device.count()
        print("SHADOW")
    """)
    assert r.returncode == 0 and "SHADOW" in r.stdout, r.stderr[-1500:]


@pytest.mark.parametrize("driver_libs", [[_LOADS], []], ids=["a-driver-that-loads", "a-table-with-no-driver-library"])
def test_a_refusal_with_a_driver_or_no_driver_library_named_stands(driver_libs):
    r = _refusing_runtime(driver_libs, """
        try:
            device.install_shadow(1)
        except DeviceQueryError as e:
            assert "error 35" in str(e), e
            print("REFUSED")
    """)
    assert r.returncode == 0 and "REFUSED" in r.stdout, (r.stdout[-500:], r.stderr[-1500:])
