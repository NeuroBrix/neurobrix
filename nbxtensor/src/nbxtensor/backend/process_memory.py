"""What THIS process holds on a device, as the driver attributes it to our context — never the card's free/total, which
counts every other process (the desktop, a browser, another model) as ours (the supervisor 2026-11-02 09:50: a profile
fact a neighbour can falsify is not a fact; runtime_growth.json for Allegro-TI2V read 7 430 MB with another run on the
card, 654 MB alone).

CUDA: NVML's per-process accounting (`nvmlDeviceGetComputeRunningProcesses_v3`, `usedGpuMemory` of our pid), the device
found by its PCI bus id (`cudaDeviceGetPCIBusId` of the process's ordinal — NVML numbers the physical cards, the runtime
the visible ones). A backend without such an accounting (HIP, Metal) or a driver that does not report it answers None:
the caller then refuses to measure, it never falls back to the card's reading.

ctypes on the driver's management library; no torch, no vendor package.
"""
from __future__ import annotations

import ctypes
import os
from typing import Optional

_NVML = None


class _ProcInfo(ctypes.Structure):
    _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong),
                ("gpuInstanceId", ctypes.c_uint), ("computeInstanceId", ctypes.c_uint)]


def _nvml(libs):
    global _NVML
    if _NVML is None:
        _NVML = False
        for name in libs or ():
            try:
                lib = ctypes.CDLL(name)
            except OSError:
                continue
            if lib.nvmlInit_v2() == 0:
                _NVML = lib
                break
    return _NVML or None


def process_device_bytes(device: int) -> Optional[int]:
    """The bytes the driver attributes to this process on the visible `device`, or None when the backend or the driver
    gives no per-process accounting."""
    from nbxtensor.backend import runtime as _rt
    table = _rt.backend_table()
    lib = _nvml(table.get("management_libs"))
    bus_fn = table.get("pci_bus_id")
    if lib is None or not bus_fn:
        return None
    bus = ctypes.create_string_buffer(64)
    if getattr(_rt.gpu_runtime(), bus_fn)(bus, 64, int(device)) != 0:
        return None
    handle = ctypes.c_void_p()
    if lib.nvmlDeviceGetHandleByPciBusId_v2(bus, ctypes.byref(handle)) != 0:
        return None
    count = ctypes.c_uint(0)
    rc = lib.nvmlDeviceGetComputeRunningProcesses_v3(handle, ctypes.byref(count), None)
    if rc not in (0, 7):                        # 7: NVML_ERROR_INSUFFICIENT_SIZE, the count is set
        return None
    n = count.value + 8                          # room for processes starting meanwhile
    infos = (_ProcInfo * n)()
    count = ctypes.c_uint(n)
    if lib.nvmlDeviceGetComputeRunningProcesses_v3(handle, ctypes.byref(count), infos) != 0:
        return None
    me = os.getpid()
    for i in range(count.value):
        if infos[i].pid == me:
            used = int(infos[i].usedGpuMemory)
            # NVML_VALUE_NOT_AVAILABLE (all bits set): the driver does not report it (a container without the pid
            # namespace, a WDDM device) — no reading, never a guess
            return None if used >= (1 << 64) - 1 else used
    return 0                                     # no context of ours on that device yet
