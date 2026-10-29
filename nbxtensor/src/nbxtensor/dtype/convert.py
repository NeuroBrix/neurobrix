"""Host conversions between bfloat16, float16 and float32 — the one module that does them.

numpy has no bfloat16. A bfloat16 array on the host is carried as its 16-bit patterns (`uint16`), or as the 2-byte
void array a bfloat16 tensor's host copy comes back as (`<V2`, the array interface's typestr for a type numpy does
not know). Every conversion between that carrier and real numbers lives here, once: rounding float32 to bfloat16,
widening bfloat16 to float32, and narrowing bfloat16 to float16.

Pure functions of numpy arrays: no device, no global state, no thread pool of their own (R-NBXT 11). A caller that
splits a large array across workers calls them on each span with `out=`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["f32_to_bf16_bits", "bf16_bits_to_f32", "bf16_bits_to_f16", "bf16_bits", "bf16_abs_max", "saturate_scalar"]

# The quiet NaN every NaN becomes, sign dropped: PyTorch's `round_to_nearest_even`
# (torch/headeronly/util/BFloat16.h: `if (std::isnan(src)) return UINT16_C(0x7FC0);`).
_QUIET_NAN = np.uint16(0x7FC0)
# The elements one step of the rounding works on: its temporaries are one step's, never the operand's size (a 5.1 GB
# operand took four full-size uint32 temporaries and the Mac's certifier to its 14 GB guard, 2026-09-28).
_STEP = 1 << 18


def bf16_bits(a: np.ndarray) -> np.ndarray:
    """The `uint16` patterns of a bfloat16 host carrier: the array itself when it already is `uint16`, a view when
    it is the 2-byte void carrier. Anything else is refused by name."""
    a = np.asarray(a)
    if a.dtype == np.uint16:
        return a
    if a.dtype.kind == "V" and a.dtype.itemsize == 2:
        return np.ascontiguousarray(a).view(np.uint16)
    raise TypeError(f"a bfloat16 host carrier is uint16 bits or a 2-byte void array, not {a.dtype}")


def f32_to_bf16_bits(a: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
    """float32 values narrowed to bfloat16, returned as their `uint16` patterns: round to nearest, ties to even —
    the rounding `torch.Tensor.to(torch.bfloat16)` applies and the device cast applies. `+0x7FFF` alone rounds an
    exact half down; `((u >> 16) & 1)` adds the one that makes it ties-to-even. Values past the bfloat16 maximum
    become infinities; every NaN becomes the quiet NaN 0x7FC0. Streamed in fixed steps: the peak is the output plus
    one step."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    flat = a.reshape(-1)
    u = flat.view(np.uint32)
    bits = np.empty(flat.shape, dtype=np.uint16) if out is None else out.reshape(-1)
    for i in range(0, flat.size, _STEP):
        uc = u[i:i + _STEP]
        t = np.right_shift(uc, np.uint32(16))
        np.bitwise_and(t, np.uint32(1), out=t)
        np.add(t, np.uint32(0x7FFF), out=t)
        np.add(t, uc, out=t)
        np.right_shift(t, np.uint32(16), out=t)
        bits[i:i + _STEP] = t
        nan = np.isnan(flat[i:i + _STEP])
        if nan.any():
            bits[i:i + _STEP][nan] = _QUIET_NAN
    return bits.reshape(a.shape) if out is None else out


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """The exact float32 values of bfloat16 patterns (`uint16`, or the 2-byte void carrier): a bfloat16 is the top
    sixteen bits of a float32, so the widening loses nothing."""
    b = np.ascontiguousarray(bf16_bits(bits))
    return (b.astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(b.shape)


def bf16_bits_to_f16(bits: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
    """bfloat16 patterns narrowed to float16, exactly as widening to float32 then numpy's float32 -> float16 cast
    (round to nearest even; past the float16 range: infinity). `out`, when given, is a float16 array of the same
    number of elements the result is written into (a caller splitting the work across workers passes spans)."""
    src = np.ascontiguousarray(bf16_bits(bits)).reshape(-1)
    dst = np.empty(src.shape, dtype=np.float16) if out is None else out.reshape(-1)
    np.copyto(dst, (src.astype(np.uint32) << np.uint32(16)).view(np.float32), casting="unsafe")
    return dst.reshape(np.shape(bits)) if out is None else out


def bf16_abs_max(bits: np.ndarray) -> float:
    """The largest magnitude among bfloat16 patterns, NaN ignored (an infinity is infinite): what a caller compares to
    a narrower dtype's `finite_max` to know whether the values fit it. Streamed in fixed steps."""
    flat = np.ascontiguousarray(bf16_bits(bits)).reshape(-1)
    top = 0.0
    for i in range(0, flat.size, _STEP):
        v = np.abs((flat[i:i + _STEP].astype(np.uint32) << np.uint32(16)).view(np.float32))
        v = v[~np.isnan(v)]
        if v.size:
            top = max(top, float(v.max()))
    return top


def saturate_scalar(value, dtype):
    """A Python number clamped to the finite range of a half floating `dtype` (a name or an enum member; the range is
    the dtype table's `max`), anything else unchanged: a bool, a non-number, a dtype wider than two bytes or not
    floating, a NaN, an infinity the graph wrote on purpose. A mask sentinel written for one half format then holds in
    the other (bfloat16's -3.39e38 becomes float16's -65504)."""
    from nbxtensor.dtype.itemsize import canonical_name, finite_max, itemsize, kind
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    name = canonical_name(dtype)
    if kind(name) != "float" or itemsize(name) != 2:
        return value
    v = float(value)
    if v != v or v in (float("inf"), float("-inf")):
        return value
    top = finite_max(name)
    return max(-top, min(top, v))
