"""Type promotion: the one rule for the dtype an op's floating inputs are brought to.

Two rules decide it, both read on dtype NAMES (plain data, R-NBXT 11) through the one table:

- `promoted(dtypes)`: an op reading floating inputs of different widths reads them all at the WIDEST, when that
  widest is wider than two bytes; among two-byte floats alone nothing is promoted (which half dtype an op computes
  in is the caller's precision policy, not a promotion).
- `binary_result(a, b)`: an element-wise op of two tensors computes in the higher of the two by kind, then by
  width; two different dtypes of one kind and width (float16 and bfloat16) keep the first operand's.
- `float_math_input(dtype)`: a floating-math op (rsqrt, exp, log ...) given an integer or bool input reads it as
  float32, as PyTorch's type promotion does; a floating or complex input is read as it is. Without it rsqrt of an
  int64 2 truncated 0.7071 to 0 (Kokoro's vocoder residual collapsed to zeros in NeuroBrix, 2026).
"""
from __future__ import annotations

from typing import Optional, Sequence

from .itemsize import itemsize, kind

__all__ = ["promoted", "float_math_input", "binary_result", "scalar_result"]

# The order of the kinds a binary result climbs (PyTorch type promotion: bool < unsigned < signed < float < complex).
_KIND_ORDER = ("bool", "uint", "int", "float", "complex")


def promoted(dtypes: Sequence[str]) -> Optional[str]:
    """The dtype every floating input of an op is brought to, or None when no promotion applies: the widest
    floating dtype among `dtypes` (the first of them in order on a tie), when it is wider than two bytes."""
    widest, size = None, 0
    for d in dtypes:
        if kind(d) != "float":
            continue
        s = itemsize(d)
        if s > size:
            widest, size = d, s
    return widest if widest is not None and size > 2 else None


def float_math_input(dtype: str) -> str:
    """The dtype a floating-math op reads an input of `dtype` at: float32 for an integer or bool input, the input's
    own dtype otherwise."""
    return dtype if kind(dtype) in ("float", "complex") else "float32"


def _of_kind(k: str, nbytes: int) -> str:
    """The table's dtype of kind `k` and `nbytes` bytes (the standard formats: intN, uintN, floatN, complexN)."""
    name = {"int": "int", "uint": "uint", "float": "float", "complex": "complex"}[k] + str(8 * nbytes)
    if kind(name) != k or itemsize(name) != nbytes:
        raise ValueError(f"no {k} dtype of {nbytes} bytes in the dtype table")
    return name


def binary_result(a: str, b: str) -> str:
    """The dtype an element-wise op of two tensors of dtypes `a` and `b` computes and writes in — the Array API
    standard's type promotion (data-apis.org/array-api, "Type Promotion Rules"), the same whatever the operands' order:
    within a kind the wider; a signed with an unsigned integer, the signed integer wide enough for both (none holds
    uint64: refused by name); a real float with a complex, the complex carrying the wider float's precision. What the
    standard leaves unspecified follows PyTorch, the engine's oracle: across kinds the higher kind (bool, unsigned,
    signed, float, complex) with its own width; two floats of one width and different formats (float16, bfloat16)
    meet in float32."""
    ka, kb = kind(a), kind(b)
    if ka == kb:
        if itemsize(a) != itemsize(b):
            return a if itemsize(a) > itemsize(b) else b
        if a == b or ka not in ("float", "complex"):
            return a
        return _of_kind(ka, 2 * itemsize(a))                 # float16 with bfloat16: float32
    if {ka, kb} == {"int", "uint"}:
        signed, unsigned = (a, b) if ka == "int" else (b, a)
        if itemsize(unsigned) >= 8:
            raise ValueError(f"binary_result({a}, {b}): no signed integer holds every {unsigned} value")
        return _of_kind("int", max(itemsize(signed), 2 * itemsize(unsigned)))
    if {ka, kb} == {"float", "complex"}:
        real, cplx = (a, b) if ka == "float" else (b, a)
        return _of_kind("complex", max(itemsize(cplx), 2 * itemsize(real)))
    return a if _KIND_ORDER.index(ka) > _KIND_ORDER.index(kb) else b


def _int_bounds(dtype: str):
    bits = 8 * itemsize(dtype)
    return (0, (1 << bits) - 1) if kind(dtype) == "uint" else (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)


def scalar_result(dtype: str, scalar) -> str:
    """The dtype an element-wise op of an array of `dtype` and a Python scalar computes in: the scalar is WEAK, it takes
    the array's dtype (the Array API standard's "mixing arrays with Python scalars", NumPy's NEP 50) — a bool with a
    bool array, an int with an integer array when the value fits it (refused by name otherwise: the standard leaves
    the overflow open, a silent wrap is never taken), an int or a float with a real float array, an int, float or
    complex with a complex array, a complex with a real float array as the complex of that precision (complex64 for a
    half float: the table holds no complex32). Left open by the
    standard and decided by NEP 50: a float with an integer or bool array is float64, a complex with one is
    complex128, an int with a bool array is int64 (PyTorch differs on the first: float32; README, Array API)."""
    k = kind(dtype)
    if isinstance(scalar, bool):
        return dtype
    if isinstance(scalar, int):
        if k in ("int", "uint"):
            lo, hi = _int_bounds(dtype)
            if not lo <= scalar <= hi:
                raise OverflowError(f"scalar_result: {scalar} does not fit {dtype} [{lo}, {hi}]")
            return dtype
        return "int64" if k == "bool" else dtype
    if isinstance(scalar, float):
        return dtype if k in ("float", "complex") else "float64"
    if isinstance(scalar, complex):
        if k == "complex":
            return dtype
        if k != "float":
            return "complex128"
        # the complex of the float's precision; a half float has none in the table (no complex32): complex64 holds it
        return _of_kind("complex", max(2 * itemsize(dtype), itemsize("complex64")))
    raise TypeError(f"scalar_result: {type(scalar).__name__} is not a Python scalar (bool, int, float, complex)")
