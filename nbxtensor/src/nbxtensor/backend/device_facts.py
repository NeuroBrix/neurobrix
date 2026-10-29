"""Device facts the library reads itself, so it runs with no application around it (API.md section 11).

Each answers from the device or from the backend's own documented capability — never from an application's profile.
An application may still provide its own answer through `nbxtensor.host` (NeuroBrix provides the one its hardware
profile declares); with none provided, the library's reading here is the answer.

- `host_shares_memory_with_device()`: Metal asks the device (`MTLDevice.hasUnifiedMemory`); CUDA asks the runtime
  (`cudaDeviceGetAttribute(cudaDevAttrIntegrated)`, attribute 18 of the CUDA runtime API's `cudaDeviceAttr`). A
  backend whose answer is not read here is refused by name.
- `kernels_carry_fp64()`: whether the backend's kernels compute float64 — Metal: no (the Metal Shading Language has
  no double type); CUDA and HIP: yes (every NVIDIA and AMD GPU computes it, at whatever rate).
- `certified_dir()`: the default store of certified kernel settings: one directory per user in the platform's cache
  (`$XDG_CACHE_HOME`, else `~/.cache`), `nbxtensor/certified/` — empty until the autotune fills it.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

__all__ = ["host_shares_memory_with_device", "kernels_carry_fp64", "certified_dir"]

_CUDA_DEV_ATTR_INTEGRATED = 18      # cudaDevAttrIntegrated (CUDA Runtime API, enum cudaDeviceAttr)
_FP64 = {"cuda": True, "hip": True, "metal": False}


def host_shares_memory_with_device() -> bool:
    """Does the current device draw from the host's memory pool?"""
    from . import runtime as _rt
    name = _rt._detect_gpu_backend()
    if name == "metal":
        return bool(_rt._gpu_runtime().has_unified_memory)
    if name == "cuda":
        rt = _rt._gpu_runtime()
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
    name = _rt._detect_gpu_backend()
    if name not in _FP64:
        raise RuntimeError(f"ZERO FALLBACK: nbxtensor does not know whether {name!r} kernels compute float64")
    return _FP64[name]


def certified_dir() -> Path:
    """The library's own store of certified kernel settings, in the user's cache."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "nbxtensor" / "certified"
