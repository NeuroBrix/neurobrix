"""A graph constant's stored bytes as the host array a tensor is built from — no tensor library opens them.

A built graph embeds its small constants (a rotary table, a window, a frequency grid) as the archive `torch.save`
writes: a ZIP whose `archive/data/<n>` members are the raw storages. `archived_storage` reads the first storage out of
it; `constant_array` turns stored bytes into the array the device receives, in the dtype the engine binds the
constant in (`dtype.operands.constant_load_dtype`).
"""
from __future__ import annotations

import io
import zipfile
from typing import Optional, Sequence, Tuple

import numpy as np

from nbxtensor.dtype import itemsize as _isz
from nbxtensor.dtype.convert import bf16_bits_to_f16, bf16_bits_to_f32
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.dtype.operands import constant_load_dtype

__all__ = ["archived_storage", "constant_array"]


def archived_storage(blob: bytes) -> Optional[bytes]:
    """The bytes of the first storage of a `torch.save` archive (its first `archive/data/` member); None when the
    archive holds no storage."""
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = [n for n in zf.namelist() if n.startswith("archive/data/")]
        return zf.read(names[0]) if names else None


def constant_array(raw: bytes, stored: str, shape: Sequence[int], compute: str,
                   stores_fp64: bool) -> Tuple[np.ndarray, Optional[NBXDtype]]:
    """`raw` (elements of the `stored` dtype, row-major `shape`) as the contiguous host array of the dtype
    `constant_load_dtype(stored, compute, stores_fp64)` names, and the dtype its BITS are when numpy cannot name it
    (bfloat16 kept as bfloat16 travels as uint16: pass it to `NBXTensor.from_numpy(dtype=)`), else None.

    bfloat16 is widened exactly (it is the top half of a float32) or narrowed to float16; it is never handed over as
    the uint16 it travels in under another dtype's name: a float32 engine then read each PAIR of bf16 values as one
    float (TinyLlama's rotary table, 2026-09-07: its first row of ones arrived as 1.0019378 = 0x3f803f80). A dtype
    numpy cannot name (float8) is refused by numpy itself, never read as float32 bytes."""
    shape = tuple(int(d) for d in shape)
    held = constant_load_dtype(stored, compute, stores_fp64)
    if stored == "bfloat16":
        bits = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
        if held == "bfloat16":
            return np.ascontiguousarray(bits), NBXDtype.bfloat16
        return (bf16_bits_to_f16(bits) if held == "float16" else bf16_bits_to_f32(bits)), None
    np_dt = np.dtype(stored)
    if np_dt.itemsize != _isz.itemsize(stored):
        raise RuntimeError(f"constant_array: numpy's {stored} is {np_dt.itemsize} bytes, the dtype table says "
                           f"{_isz.itemsize(stored)}")
    arr = np.frombuffer(raw, dtype=np_dt).reshape(shape)
    return np.ascontiguousarray(arr if held == stored else arr.astype(np.dtype(held))), None
