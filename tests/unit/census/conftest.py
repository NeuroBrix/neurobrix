"""Census cells bind a compile target for the process; none may leave it bound.

`neurobrix.kernels.census._bind_target` declares a hardware profile's compile target to the library
(`nbxtensor.device.declare_target` -> `nbxtensor.backend.launcher.bind_target`) and forgets the device
description in force. The census process is one-shot, so the bind stays; a pytest process is not. Two cells
bound a V100 target and every kernel test after them in the same process compiled for it, ran the x86 NVIDIA
ptxas on a Mac and failed with "Exec format error" (about 412 reds in an in-order run, green alone).
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def census_target_is_scoped():
    """Save the process's compile target before each census cell and restore it after, whatever the outcome.

    The restore goes through the library's own door (`launcher.bind_target`, which accepts the unbound state
    and so hands the answer back to the host's driver) and the device description is read again under it."""
    from nbxtensor.backend import device_files, launcher
    before = launcher._TARGET
    try:
        yield
    finally:
        launcher.bind_target(before)
        device_files.forget()
