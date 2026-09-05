"""Which driver the launcher uses — the one place a backend is named.

`launcher.py` is a single vendor-agnostic engine component: it binds
arguments, recomputes the specialization, and asks *the registered driver*
to compile and dispatch. It never asks what that driver is. That question is
answered here, once, from a table.

The table is data, not a branch. Adding ROCm is a row; adding CUDA is a row.
Neither touches the launcher, the wrappers, or a kernel.

CUDA has no row yet, deliberately. The machine that owns CUDA has not
activated a NeuroBrix driver and has not re-measured its zoo, so on that
machine `activate()` registers nothing, the launcher stays transparent, and
`kernel[grid](...)` resolves exactly as it did before this component existed
(`test_launcher_is_transparent_without_a_driver`). Activating CUDA is one
row plus that machine's measurements — not a rewrite.

The backend name itself is not decided here either: it comes from the seam
that already resolves it for the allocator (`_detect_gpu_backend`). One
detection for the whole engine; a second one would be a second opinion.
"""

from __future__ import annotations

import os
import threading
from importlib import import_module
from typing import Optional

from . import launcher

#: backend name (as the allocator seam resolves it) -> driver module.
#: The module must expose `driver()` returning a `LauncherDriver`
#: (`neurobrix.triton.launcher_contract`), which `verify_driver_contract`
#: checks for real.
_DRIVER_MODULES = {
    "metal": "neurobrix.triton.metal_driver",
}

#: Measurement switch, not a fallback. Set to "triton" to leave the launcher
#: uninstalled and run Triton's own `kernel[grid]`, which is how the two
#: paths are compared on the same machine ("first light identical under the
#: new launcher"). Any other value, or unset, means the engine's launcher.
#: It cannot rescue a broken driver: with a driver registered and a kernel
#: that refuses, the refusal is loud either way.
_ENV_SWITCH = "NBX_LAUNCHER"

_LOCK = threading.Lock()
_ACTIVATED: Optional[str] = None


def driver_module_for(backend: str) -> Optional[str]:
    """The driver module registered for `backend`, or None."""
    return _DRIVER_MODULES.get(backend)


def activate(backend: Optional[str] = None) -> Optional[str]:
    """Install the launcher and register the driver for the active backend.

    Returns the backend whose driver was registered, or None when there is
    none — in which case nothing is installed and Triton's path is untouched.
    Idempotent: the engine calls it at import of the dispatch layer.
    """
    global _ACTIVATED

    if os.environ.get(_ENV_SWITCH, "").strip().lower() == "triton":
        return None

    with _LOCK:
        if _ACTIVATED is not None:
            return _ACTIVATED

        if backend is None:
            from .nbx_tensor import _detect_gpu_backend
            try:
                backend = _detect_gpu_backend()
            except Exception:
                # No GPU resolved at all. Not this module's failure to
                # report: the allocator raises for real when someone asks
                # it for a device. Here it simply means no driver.
                return None

        module_path = _DRIVER_MODULES.get(backend)
        if module_path is None:
            return None

        module = import_module(module_path)
        launcher.register_driver(module.driver())
        launcher.install()
        _ACTIVATED = backend
        return backend


def deactivate() -> None:
    """Unregister and uninstall. For tests and for A/B measurement."""
    global _ACTIVATED
    with _LOCK:
        launcher.unregister_driver()
        launcher.uninstall()
        _ACTIVATED = None


def activated_backend() -> Optional[str]:
    return _ACTIVATED
