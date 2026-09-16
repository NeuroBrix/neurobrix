"""Suite-wide conftest.

WHY THIS FILE EXISTS
--------------------
The workshop runs gates under a DOOR — `CUDA_VISIBLE_DEVICES=` — so a suite can
run beside a timed campaign without a context ever existing on a real card
(`docs/reference/proving-by-doors.md`). Measured on 2026-09-16, the unit suite
under that door returned **50 failures, 42 of them one thing**: a test that
allocates on a device that is not there, raising

    DeviceOOMError: GPU malloc failed (error 100) for 64 bytes
    [device cuda:0 ... driver_free=0MB / driver_total=0MB]

error 100 being `cudaErrorNoDevice`. A test that cannot run must SAY so, not
fail; fifty reds that mean "no card here" are fifty reds nobody reads, and they
are what makes a suite stop being run beside a campaign at all.

WHY IT IS NOT A SWALLOWED ERROR
-------------------------------
The conversion happens only when an EXECUTING probe says this process can see no
device at all — `DeviceAllocator.device_count()`, which asks the driver, not
`_detect_gpu_backend()`, which answers which backend the BUILD can address and
says "cuda" with no card visible (register 62). On a machine with a device the
hook does nothing whatsoever: it is disarmed for the session, so a real
allocation failure on a real card fails exactly as it does today. And it
converts only a failure whose own driver report says the device has zero total
memory — a genuine out-of-memory on a live card names a real total.

Adding a guard to each of the forty-two would have been the same line written
forty-two times, and the forty-third test would have been written without it.
"""
from __future__ import annotations

import pytest

# Each vendor phrases the same fact its own way; torch's is the last one, and it
# reached the suite through a bridge test rather than through the allocator.
_NO_DEVICE_MARKS = ("error 100", "cudaErrorNoDevice", "driver_total=0MB",
                    "No GPU runtime found", "No CUDA GPUs are available")


def _devices_visible() -> int:
    """How many devices the DRIVER reports, asked once, by executing the question."""
    try:
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        return int(DeviceAllocator.device_count())
    except Exception:
        return 0


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """pytest 9's wrapper form: the test's exception arrives here to be re-raised."""
    try:
        return (yield)
    except Exception as exc:                              # noqa: BLE001 — re-raised below
        if _devices_visible() > 0:
            raise                   # a real machine: this hook is not armed at all
        text = f"{type(exc).__name__}: {exc}"
        if any(m in text for m in _NO_DEVICE_MARKS):
            pytest.skip(f"no GPU visible to this process — {type(exc).__name__} at the "
                        f"first allocation, which is the absence of a card and not a "
                        f"result: {str(exc)[:160]}")
        raise
