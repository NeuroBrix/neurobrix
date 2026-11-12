"""A tensor of another library taken in through DLPack, the one exchange protocol nbxtensor speaks (DESIGN section 3:
"numpy and DLPack, never torch").

Any object with `__dlpack__` (a torch tensor, a numpy array, ...) hands over a capsule holding a `DLManagedTensor`;
it is read here with ctypes, so nothing of the producer is imported. What the capsule says decides the path:

* host memory (kDLCPU): the elements are copied onto the device, its strides honoured (a channels-last view arrives
  as the values it shows, not as its buffer's order) and bfloat16 carried as its bits;
* this backend's own device memory (kDLCUDA on CUDA, kDLROCM on HIP): wrapped where it is, no copy, the producer's
  memory kept alive for as long as the tensor lives; a strided one is made contiguous by the library's own kernel;
* anything else (another vendor's memory: an MPS tensor's buffer is torch's on Metal, not this runtime's) is refused
  by name: the caller hands over host memory instead.

Before (the audit of 2026-11-12, TENSOR_IN_ENGINE_A rows 93 and 95): the engine carried four copies of a torch to
NBXTensor conversion: a CPU bfloat16 tensor failed in all four (`.numpy()`: numpy has no bfloat16), an MPS tensor in
two, and on a CUDA card one wrapped a non-contiguous tensor's pointer with contiguous strides.
"""
from __future__ import annotations

import ctypes
from typing import Any, Optional, Tuple

_CPU, _CUDA, _CUDA_HOST, _ROCM = 1, 2, 3, 10
#: the device types whose memory a backend dispatches on directly (torch reports a ROCm card as kDLROCM)
_OWN_DEVICE = {"cuda": _CUDA, "hip": _ROCM}


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", _DLDevice), ("ndim", ctypes.c_int32),
                ("dtype", _DLDataType), ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)), ("byte_offset", ctypes.c_uint64)]


class _DLManagedTensor(ctypes.Structure):
    #: the deleter is `void (*)(DLManagedTensor *)`, held as an address and called through `_DELETER`
    _fields_ = [("dl_tensor", _DLTensor), ("manager_ctx", ctypes.c_void_p), ("deleter", ctypes.c_void_p)]


_DELETER = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

_capsule_pointer = ctypes.pythonapi.PyCapsule_GetPointer
_capsule_pointer.restype, _capsule_pointer.argtypes = ctypes.c_void_p, [ctypes.py_object, ctypes.c_char_p]
_capsule_rename = ctypes.pythonapi.PyCapsule_SetName
_capsule_rename.restype, _capsule_rename.argtypes = ctypes.c_int, [ctypes.py_object, ctypes.c_char_p]

#: (DLPack type code, bits) -> (NBXDtype name, numpy container of the same width)
_DTYPES = {(2, 16): ("float16", "float16"), (4, 16): ("bfloat16", "uint16"), (2, 32): ("float32", "float32"),
           (2, 64): ("float64", "float64"), (0, 8): ("int8", "int8"), (0, 16): ("int16", "int16"),
           (0, 32): ("int32", "int32"), (0, 64): ("int64", "int64"), (1, 8): ("uint8", "uint8"),
           (6, 8): ("bool_", "bool"), (5, 64): ("complex64", "complex64"), (5, 128): ("complex128", "complex128")}


class _Owner:
    """Keeps the producer's memory: calls the capsule's deleter once, when the last tensor reading it is gone."""

    def __init__(self, managed: Any, producer: Any):
        self._managed, self._producer = managed, producer

    def release(self) -> None:
        managed, self._managed = self._managed, None
        if managed and managed.contents.deleter:
            _DELETER(managed.contents.deleter)(ctypes.cast(managed, ctypes.c_void_p))

    def __del__(self):
        self.release()


def _taken(obj: Any) -> Tuple[_Owner, _DLTensor]:
    """The capsule's tensor, the capsule marked used (DLPack: a consumer takes it once)."""
    capsule = obj.__dlpack__()
    address = _capsule_pointer(capsule, b"dltensor")
    if not address:
        raise RuntimeError(f"from_dlpack: {type(obj).__name__}.__dlpack__() gave no unused 'dltensor' capsule")
    _capsule_rename(capsule, b"used_dltensor")
    managed = ctypes.cast(address, ctypes.POINTER(_DLManagedTensor))
    return _Owner(managed, obj), managed.contents.dl_tensor


def _layout(t: _DLTensor, what: str):
    """(NBXDtype name, numpy container, shape, element strides or None for C order)."""
    if t.dtype.lanes != 1:
        raise TypeError(f"from_dlpack: {what} has {t.dtype.lanes} lanes per element; only scalar elements are read")
    kind = _DTYPES.get((t.dtype.code, t.dtype.bits))
    if kind is None:
        raise TypeError(f"from_dlpack: {what} holds elements of DLPack type code {t.dtype.code}, {t.dtype.bits} bits, "
                        f"which nbxtensor has no dtype for")
    shape = tuple(int(t.shape[i]) for i in range(t.ndim))
    strides = tuple(int(t.strides[i]) for i in range(t.ndim)) if t.strides else None
    return kind[0], kind[1], shape, strides


def _is_c_order(shape, strides) -> bool:
    if strides is None:
        return True
    want = 1
    for n, s in zip(reversed(shape), reversed(strides)):
        if n != 1 and s != want:
            return False
        want *= n
    return True


def _from_host(t: _DLTensor, owner: _Owner, dtype: str, container: str, shape, strides, device):
    import numpy as np
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.tensor import NBXTensor
    item = np.dtype(container).itemsize
    count = 1
    for n in shape:
        count *= n
    if count == 0:
        out = NBXTensor.from_numpy(np.zeros(shape, dtype=container), dtype=getattr(NBXDtype, dtype), device=device)
        owner.release()
        return out
    strides = strides or tuple(int(np.prod(shape[i + 1:], dtype=np.int64)) for i in range(len(shape)))
    span = 1 + sum((n - 1) * abs(s) for n, s in zip(shape, strides))
    start = int(t.data) + int(t.byte_offset) + sum((n - 1) * s for n, s in zip(shape, strides) if s < 0) * item
    flat = np.ctypeslib.as_array((ctypes.c_uint8 * (span * item)).from_address(start)).view(container)
    first = -sum((n - 1) * s for n, s in zip(shape, strides) if s < 0)
    view = np.lib.stride_tricks.as_strided(flat[first:], shape=shape, strides=tuple(s * item for s in strides))
    host = np.ascontiguousarray(view)                   # the values the producer shows, in C order: a copy
    owner.release()
    return NBXTensor.from_numpy(host, dtype=getattr(NBXDtype, dtype), device=device)


def from_dlpack(obj: Any, device: Optional[Any] = None):
    """An NBXTensor of the elements `obj` exports through DLPack (`obj.__dlpack__()`). Host memory is copied onto
    `device` (an index or "cuda:N"; the current device when None, as `NBXTensor.from_numpy` places); this backend's
    own device memory is wrapped in place, the producer's memory kept alive with it. Another vendor's device memory
    is refused by name."""
    from nbxtensor.backend.runtime import gpu_backend
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.tensor import NBXTensor
    owner, t = _taken(obj)
    what = f"a {type(obj).__name__} on DLPack device {t.device.device_type}:{t.device.device_id}"
    dtype, container, shape, strides = _layout(t, what)
    if t.device.device_type in (_CPU, _CUDA_HOST):
        return _from_host(t, owner, dtype, container, shape, strides, device)
    if t.device.device_type != _OWN_DEVICE.get(gpu_backend()):
        owner.release()
        raise RuntimeError(f"from_dlpack: {what} is memory this {gpu_backend()} runtime does not own; hand over host "
                           f"memory (a host copy of it) instead")
    address, card = int(t.data) + int(t.byte_offset), int(t.device.device_id)
    if strides is None or _is_c_order(shape, strides):
        return NBXTensor.from_raw(address, shape, getattr(NBXDtype, dtype), "cuda", owns_data=False, device_idx=card,
                                  base=owner)
    view = NBXTensor(address, shape, strides, getattr(NBXDtype, dtype), "cuda", owns_data=False, device_idx=card,
                     base=owner)
    return view.contiguous()                            # the library's own strided copy, on the card
