"""A tensor's values read on the host for a diagnostic: non-finite scan, value statistics, the last position's row, a
digest of the bytes, a few values at row-major positions.

What an engine prints or writes when it inspects an op's output (which op, which tensor, which file) is its own
choice; reading the values is the library's. Every function copies to the host through `NBXTensor.numpy()` (floats
cast to float32 on the device first, a complex tensor read as its real view: real and imaginary parts interleaved)
and writes nothing to the device. Torch-free.
"""
from __future__ import annotations

import hashlib
import math
from typing import Optional, Tuple

import numpy as np

from nbxtensor.dtype.nbx_dtype import NBXDtype

__all__ = ["host_values", "non_finite", "value_stats", "last_position_row", "digest", "values_at"]


def _real(t):
    return t.view_as_real() if t.is_complex() else t


def host_values(t) -> np.ndarray:
    """Every value of `t` as one flat float32 host array, a complex tensor as its interleaved real and imaginary
    parts."""
    return _real(t).to(NBXDtype.float32).numpy().reshape(-1)


def non_finite(t) -> Optional[str]:
    """"NaN" when a value of `t` is NaN, else "Inf" when one is infinite, else None (a complex tensor: both parts)."""
    v = host_values(t)
    if np.isnan(v).any():
        return "NaN"
    return "Inf" if np.isinf(v).any() else None


def _last_position(shape) -> Optional[Tuple[int, int]]:
    """(offset, length) of the last position's row in a row-major array of `shape`: index -1 along axis 1 at rank 3
    or more (batch 0), along axis 0 at rank 2; None below rank 2."""
    if len(shape) >= 3:
        row = math.prod(shape[2:])
        return (shape[1] - 1) * row, row
    if len(shape) == 2:
        return (shape[0] - 1) * shape[1], shape[1]
    return None


def last_position_row(t) -> Optional[np.ndarray]:
    """The whole last position's row of `t` (its real view) as float32, None below rank 2 (`value_stats`' window)."""
    at = _last_position(list(_real(t).shape))
    return None if at is None else host_values(t)[at[0]:at[0] + at[1]]


def value_stats(t) -> dict:
    """What a per-op value dump records of `t`: shape and dtype as given, whether it is complex, `head10` (the first
    ten values of the first row), `last_pos10` (the first ten of the last position's row, None below rank 2),
    `l2_norm` (accumulated in float64), `batch_norms` (each batch item's L2 when the batch is 2 or 3, else None). A
    complex tensor is read as its real view. A tensor with no element is recorded without reading it."""
    shape = list(t.shape)
    rec = {"shape": shape, "dtype": str(t.dtype), "is_complex": bool(t.is_complex()), "head10": [],
           "last_pos10": None, "l2_norm": 0.0, "batch_norms": None}
    if t.numel() == 0:
        return rec
    real_shape = list(_real(t).shape)
    vals = host_values(t)
    v64 = vals.astype(np.float64)
    rec["head10"] = vals[:min(10, real_shape[-1] if real_shape else 1)].tolist()
    rec["l2_norm"] = float(np.sqrt(np.dot(v64, v64)))
    at = _last_position(real_shape)
    if at is not None:
        rec["last_pos10"] = vals[at[0]:at[0] + min(10, at[1])].tolist()
    if len(shape) >= 2 and shape[0] in (2, 3) and vals.size % shape[0] == 0:
        per = vals.size // shape[0]
        rec["batch_norms"] = [float(np.sqrt(np.dot(v64[b * per:(b + 1) * per], v64[b * per:(b + 1) * per])))
                              for b in range(shape[0])]
    return rec


def digest(t, algorithm: str, nbytes: Optional[int] = None, **params) -> str:
    """The hex digest (`hashlib.new(algorithm, ..., **params)`) of `t`'s bytes in row-major order, the first `nbytes`
    of them when given."""
    data = t.numpy().tobytes()
    return hashlib.new(algorithm, data if nbytes is None else data[:nbytes], **params).hexdigest()


def values_at(t, positions) -> list:
    """The values of `t` at the row-major flat `positions`, as Python floats, each element read alone (the tensor is
    never copied whole: a sample of a large output stays cheap). The device is synchronised first, so a kernel still
    writing `t` is waited for."""
    from nbxtensor.memory.allocator import DeviceAllocator
    DeviceAllocator.sync_device()
    shape = list(t.shape)
    out = []
    for p in positions:
        coords = []
        for size in reversed(shape):
            coords.append(p % size)
            p //= size
        v = t
        for i in reversed(coords):
            v = v.select(0, i)
        out.append(float(v.item()))
    return out


def label(t) -> str:
    """A tensor as a diagnostic line names it: shape, dtype, device and this object's id (stable while it lives),
    never its address ("-" for an empty slot)."""
    if t is None:
        return "-"
    return f"{tuple(t.shape)} {t.nbx_dtype.name} {'cpu' if t.is_cpu else 'device'}:{t.device_idx} id={id(t):#x}"
