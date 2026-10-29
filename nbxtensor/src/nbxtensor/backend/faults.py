"""The device-fault channel: a kernel on a backend that does not honour `tl.device_assert` reports a contract violation
through a device word, read at the sync points. Cut out of tensor.py (M2, 2026-10-29), unchanged."""

from __future__ import annotations

import ctypes
import functools
import threading

from nbxtensor.backend import capabilities as _caps


# Whether this backend's device memory is directly readable at its own
# `data_ptr()` from the host. The fault channel below reads its status word
# that way — after a flush that has already happened, so no copy and no
# recursion through `memcpy`. Unified memory says yes; a discrete heap does
# not, and must add its row (and a copy path) before the channel is armed
# on it.
_BACKEND_MEMORY_IS_HOST_READABLE = {"cuda": False, "hip": False, "metal": True}


# ============================================================================
# DEVICE-SIDE FAULT CHANNEL
# ============================================================================
#
# The portable half of `tl.device_assert`, for backends that do not honour
# it. A kernel that detects a broken contract stores its fault code into a
# one-word status buffer; the code is a `tl.constexpr`, so on a backend that
# DOES trap the wrapper passes 0 and not one instruction is emitted (the
# CUDA path is unchanged down to the generated PTX bar one unused kernel
# parameter).
#
# Why a device-side word rather than checking the indices on the host:
# measured 2026-09-10 on this machine, a host readback of the index tensor
# costs +0.24 ms (1 id) to +0.29 ms (512 ids) per call against an embedding
# launch of 0.017–0.020 ms — 13x to 17x the op it guards, and flat in the
# number of indices because the cost is the `flush()` a D2H copy performs,
# not the data. That is the same trap the D2D measurement of 2026-09-07
# found (1,086 host-ordered copies in an 8-token decode: 0.007 s -> 4.08 s).
# The status word costs one conditional store on the failing path and a
# 4-byte read at a sync that was already happening.
#
# It is READ where the host observes device results — after `sync_device()`
# and after a copy to the host — which is the sticky-error contract
# `sync_device` already documents: the fault is at or before that point.
_FAULT_MESSAGES: list = []
_FAULT_BUFFERS: dict = {}
_FAULT_LOCK = threading.Lock()


def register_device_fault(message: str) -> int:
    """Reserve a fault code for `message`. Called at import by the kernel
    that raises it, so codes are stable within a process and never zero
    (zero is 'clean', and it is what a disarmed kernel is compiled with)."""
    with _FAULT_LOCK:
        if message not in _FAULT_MESSAGES:
            _FAULT_MESSAGES.append(message)
        return _FAULT_MESSAGES.index(message) + 1


def device_fault_code(message: str, *, armed: bool = None) -> int:
    """The code a kernel should be compiled with: its own where the backend
    needs the channel, 0 where the backend traps by itself."""
    if armed is None:
        armed = not _caps.backend_traps_on_device_assert()
    return register_device_fault(message) if armed else 0


@functools.lru_cache(maxsize=None)
def device_fault_code_cached(message: str) -> int:
    """`device_fault_code` memoised per message — it runs at every launch of
    an armed kernel, and the answer cannot change within a process."""
    return device_fault_code(message)


def fault_channel(message: str, spare: 'NBXTensor') -> tuple:
    """`(pointer_argument, FAULT_CODE)` for a kernel that carries the channel.

    Where the backend HONOURS `tl.device_assert`, the code is 0 and the
    generated kernel contains no fault store at all — proven on the emitted
    code, not assumed: with FAULT_CODE=0 the body carries neither the store nor
    the reduction that feeds it. The pointer is then a bound-but-unread
    argument, so `spare` — a tensor the call already owns — is handed over and
    NOTHING IS ALLOCATED.

    That matters because the fault buffer is never freed by contract (a frozen
    replay plan records its raw pointer), so allocating it on a backend that
    can never write to it is a permanent allocation for a dead path. Reported
    from the other machine 2026-09-11, whose CUDA proof also says the guard
    itself holds: three kernels refuse an out-of-range index by name, six
    outputs identical, +0.13% wall.

    `spare` is never dereferenced through this argument on that path; passing
    the output tensor is therefore safe and costs nothing.
    """
    code = device_fault_code_cached(message)
    if code == 0:
        return spare, 0
    return device_fault_buffer(spare._device_idx), code


def device_fault_buffer(device_idx: int) -> 'NBXTensor':
    """This device's zeroed one-word status buffer.

    Handed to the kernel as a TENSOR, not as a raw address: the launcher
    binds a tensor as a pointer argument, and a bare int arrives as an
    int64 scalar that `tl.store` refuses.

    Never freed: a kernel launch records this raw pointer, and a frozen
    replay plan may hold it long after the call that produced it — the same
    reason `_INT64_ARRAY_CACHE` is never evicted.
    """
    buf = _FAULT_BUFFERS.get(device_idx)          # hot path: no lock
    if buf is None:
        with _FAULT_LOCK:
            buf = _FAULT_BUFFERS.get(device_idx)
            if buf is None:
                from nbxtensor.dtype.nbx_dtype import NBXDtype  # the tensor module imports this one
                from nbxtensor.tensor import NBXTensor
                buf = NBXTensor.zeros((1,), NBXDtype.int32, f"cuda:{device_idx}")
                _FAULT_BUFFERS[device_idx] = buf
    return buf


def check_device_faults() -> None:
    """Raise if any kernel reported a broken contract since the last check.

    Reads the status word straight at its own address: this runs only after
    a flush (a completed sync, or a copy that flushed), and only on a
    backend whose device memory the host can read directly. The word is
    cleared as it is read, so one violation raises once.
    """
    if not _FAULT_BUFFERS:
        return
    if not _caps._backend_capability(
            _BACKEND_MEMORY_IS_HOST_READABLE, "_BACKEND_MEMORY_IS_HOST_READABLE",
            "whether the host can read its device memory directly"):
        return
    for device_idx, buf in list(_FAULT_BUFFERS.items()):
        word = ctypes.c_int32.from_address(buf.data_ptr())
        code = word.value
        if code:
            word.value = 0
            message = (_FAULT_MESSAGES[code - 1]
                       if 1 <= code <= len(_FAULT_MESSAGES)
                       else f"unregistered fault code {code}")
            raise RuntimeError(
                f"device-side contract violated on device {device_idx}: "
                f"{message}. The kernel reported it through the fault "
                f"channel because this backend does not honour "
                f"`tl.device_assert`; the fault is at or before this point.")
