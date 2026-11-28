"""GPU sticky-fault watchdog for ServingDaemon.

Detects unrecoverable GPU context faults (CUDA, HIP/ROCm, Apple Metal) where a
Python exception handler would otherwise leave a poisoned device context
resident in VRAM. On a sticky fault, logs a `[Watchdog]` diagnostic, performs
best-effort cleanup of daemon IPC files, and terminates the process with
`os._exit(1)` so the OS kernel reclaims 100% of VRAM.

Pure standard library: zero Torch or GPU driver imports.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Callable, Iterable, Optional

_log = logging.getLogger(__name__)

# Recoverable OOM errors must NEVER trigger an immediate watchdog abort; the
# serving engine handles OOM by re-planning or reporting a normal error.
_RECOVERABLE_OOM_MARKERS: tuple[str, ...] = (
    "out of memory",
    "cudaerrormemoryallocation",
    "hiperrormemoryallocation",
    "outofmemoryerror",
    "deviceoutofmemory",
)

# Substrings (lowercase) identifying unrecoverable sticky GPU faults across
# CUDA, HIP/ROCm, and Apple Metal backends.
_STICKY_GPU_FAULT_MARKERS: tuple[str, ...] = (
    # CUDA sticky context / execution faults
    "illegal memory access",
    "cudaerrorillegaladdress",
    "cuda_error_illegal_address",
    "device-side assert",
    "cudaerrorassert",
    "cuda_error_assert",
    "unspecified launch failure",
    "cudaerrorlaunchfailure",
    "cuda_error_launch_failed",
    "misaligned address",
    "cudaerrormisalignedaddress",
    "cuda_error_misaligned_address",
    "illegal instruction",
    "cudaerrorillegalinstruction",
    "cuda_error_illegal_instruction",
    "uncorrectable ecc",
    "ecc error",
    "cudaerroreccuncorrectable",
    "cuda_error_ecc_uncorrectable",
    "cuda context is destroyed",
    "cudaerrorcontextisdestroyed",
    "cuda_error_context_is_destroyed",
    "hardware stack error",
    "cudaerrorhardwarestackerror",
    "cuda_error_hardware_stack_error",
    "sticky cuda error",
    # HIP / ROCm sticky faults
    "hiperrorillegaladdress",
    "hiperrorillegalstate",
    "hiperrorassert",
    "hiperrorlaunchfailure",
    "hiperroreccuncorrectable",
    "hiperrormisalignedaddress",
    "memory access fault by gpu",
    "gpu memory access fault",
    "hip error: illegal",
    # Apple Metal sticky / command-buffer faults
    "mtlcommandbufferstatuserror",
    "mtlcommandbuffererror",
    "command buffer execution failed",
    "iogpucommandbuffer",
    "metal device lost",
    "gpu hang",
)


def _iter_exception_chain(exc: BaseException) -> Iterable[BaseException]:
    """Yield `exc` and its `__cause__` / `__context__` chain without cycles."""
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        cur = stack.pop()
        obj_id = id(cur)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        yield cur
        if cur.__cause__ is not None:
            stack.append(cur.__cause__)
        if cur.__context__ is not None:
            stack.append(cur.__context__)


def _is_recoverable_oom(text: str) -> bool:
    return any(marker in text for marker in _RECOVERABLE_OOM_MARKERS)


def is_sticky_gpu_fault(exc: BaseException) -> bool:
    """Return True if `exc` (or any chained cause/context) is an unrecoverable
    GPU sticky fault that leaves the device context poisoned."""
    for cur in _iter_exception_chain(exc):
        text = f"{type(cur).__name__}: {cur}".lower()
        if _is_recoverable_oom(text):
            continue
        if any(marker in text for marker in _STICKY_GPU_FAULT_MARKERS):
            return True
    return False


def _flush_logs() -> None:
    """Best-effort flush of logging handlers and stderr before `os._exit`."""
    for handler in list(_log.handlers) + list(logging.getLogger().handlers):
        try:
            handler.flush()
        except Exception:
            pass
    try:
        sys.stderr.flush()
    except Exception:
        pass


def abort_on_sticky_fault(
    exc: BaseException,
    cleanup_fn: Optional[Callable[[], Any]] = None,
    exit_fn: Callable[[int], Any] = os._exit,
) -> bool:
    """If `exc` is a sticky GPU fault, log `[Watchdog]`, run best-effort
    `cleanup_fn`, and immediately call `exit_fn(1)` (`os._exit(1)` by default).

    Returns False when `exc` is not a sticky GPU fault; returns True if a
    sticky fault was detected and `exit_fn(1)` returned (e.g. in unit tests).
    """
    if not is_sticky_gpu_fault(exc):
        return False

    _log.error(
        "[Watchdog] Unrecoverable sticky GPU fault detected (%s: %s) — "
        "cleaning up daemon files and terminating immediately (exit 1) so the "
        "OS kernel reclaims 100%% of VRAM.",
        type(exc).__name__,
        exc,
    )
    if cleanup_fn is not None:
        try:
            cleanup_fn()
        except Exception as cleanup_exc:
            _log.error("[Watchdog] Best-effort cleanup failed: %s", cleanup_exc)
    _flush_logs()
    exit_fn(1)
    return True
