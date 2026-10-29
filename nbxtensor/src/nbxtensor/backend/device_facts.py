"""Device facts the library reads itself, so it runs with no application around it (API.md section 11).

Each answers from the device or from the backend's own documented capability — never from an application's profile.
An application may still provide its own answer through `nbxtensor.host` (NeuroBrix provides the one its hardware
profile declares); with none provided, the library's reading here is the answer.

- `host_shares_memory_with_device()`: Metal asks the device (`MTLDevice.hasUnifiedMemory`); CUDA asks the runtime
  (`cudaDeviceGetAttribute(cudaDevAttrIntegrated)`, attribute 18 of the CUDA runtime API's `cudaDeviceAttr`). A
  backend whose answer is not read here is refused by name.
- `kernels_carry_fp64()`: whether the backend's kernels compute float64 — Metal: no (the Metal Shading Language has
  no double type); CUDA and HIP: yes (every NVIDIA and AMD GPU computes it, at whatever rate).
- `memory_gb(memory_mb)` / `device_memory_gb(device_idx)`: a card's memory rounded to GB, the size a certified kernel
  setting is proven for (an entry serves only cards of the size it was proven on: a 16 GB and a 32 GB V100 share an
  architecture and not their best tiles under memory pressure). The device's memory is the one an application declared
  (`declare_device_memory`: a census shadow sees no card; a plan made for a profile) or, with none declared, the
  driver's total, read once per device.
- `certified_dir()`: the default store of certified kernel settings: one directory per user in the platform's cache
  (`$XDG_CACHE_HOME`, else `~/.cache`), `nbxtensor/certified/` — empty until the autotune fills it.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

__all__ = ["host_shares_memory_with_device", "kernels_carry_fp64", "certified_dir", "memory_gb", "declare_device_memory",
           "device_memory_gb"]

_CUDA_DEV_ATTR_INTEGRATED = 18      # cudaDevAttrIntegrated (CUDA Runtime API, enum cudaDeviceAttr)
_FP64 = {"cuda": True, "hip": True, "metal": False}
_DECLARED_MEMORY: dict = {}          # device index -> bytes an application declared
_MEASURED_MEMORY: dict = {}          # device index -> bytes the driver reported, read once


def memory_gb(memory_mb) -> "int | None":
    """Memory in MB, rounded to GB (`round(mb / 1024)`); None when unknown or not positive."""
    try:
        mb = float(memory_mb)
    except (TypeError, ValueError):
        return None
    return int(round(mb / 1024)) if mb > 0 else None


def declare_device_memory(device_idx: int, nbytes) -> None:
    """An application states device `device_idx`'s memory in bytes (None forgets it): a census shadow, which sees no
    card, or a plan made for a profile whose card is not the one in this process."""
    if nbytes is None:
        _DECLARED_MEMORY.pop(int(device_idx), None)
    else:
        _DECLARED_MEMORY[int(device_idx)] = int(nbytes)


def declared_device_memory() -> dict:
    """The declared devices' memory, by index (a copy)."""
    return dict(_DECLARED_MEMORY)


def device_memory_gb(device_idx: int) -> "int | None":
    """Device `device_idx`'s memory in GB: the declared one, else the driver's total (read once); None when neither
    answers (no card visible and none declared)."""
    idx = int(device_idx)
    nbytes = _DECLARED_MEMORY.get(idx)
    if nbytes is None:
        if idx not in _MEASURED_MEMORY:
            from nbxtensor.memory.allocator import DeviceAllocator
            _MEASURED_MEMORY[idx] = int(DeviceAllocator.device_total_bytes(idx))
        nbytes = _MEASURED_MEMORY[idx]
    return memory_gb(nbytes / (1 << 20)) if nbytes and nbytes > 0 else None


def host_shares_memory_with_device() -> bool:
    """Does the current device draw from the host's memory pool?"""
    from . import runtime as _rt
    name = _rt.gpu_backend()
    if name == "metal":
        return bool(_rt.gpu_runtime().has_unified_memory)
    if name == "cuda":
        rt = _rt.gpu_runtime()
        rc, dev = _rt.device_memory().get_device()
        value = ctypes.c_int()
        if rc == 0:
            rc = rt.cudaDeviceGetAttribute(ctypes.byref(value), ctypes.c_int(_CUDA_DEV_ATTR_INTEGRATED),
                                           ctypes.c_int(dev))
        if rc != 0:
            raise RuntimeError(f"ZERO FALLBACK: cudaDeviceGetAttribute(cudaDevAttrIntegrated) failed (error {rc}): "
                               f"whether this device shares the host's memory is not known")
        return bool(value.value)
    raise RuntimeError(f"ZERO FALLBACK: whether a {name!r} device shares the host's memory is not read by nbxtensor; "
                       f"provide it (nbxtensor.host.provide('host_shares_memory_with_device', ...))")


def kernels_carry_fp64(profile=None) -> bool:
    """Do the backend's kernels compute float64? (`profile`: an application's hardware profile, which an application's
    own answer reads; the library's answer is the backend's.)"""
    from . import runtime as _rt
    name = _rt.gpu_backend()
    if name not in _FP64:
        raise RuntimeError(f"ZERO FALLBACK: nbxtensor does not know whether {name!r} kernels compute float64")
    return _FP64[name]


def certified_dir() -> Path:
    """The library's own store of certified kernel settings, in the user's cache."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "nbxtensor" / "certified"
