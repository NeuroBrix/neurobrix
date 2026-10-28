"""Type promotion: the one rule for the dtype an op's floating inputs are brought to.

Two rules decide it, both read on dtype NAMES (plain data, R-NBXT 11) through the one table:

- `promoted(dtypes)`: an op reading floating inputs of different widths reads them all at the WIDEST, when that
  widest is wider than two bytes; among two-byte floats alone nothing is promoted (which half dtype an op computes
  in is the caller's precision policy, not a promotion).
- `float_math_input(dtype)`: a floating-math op (rsqrt, exp, log ...) given an integer or bool input reads it as
  float32, as PyTorch's type promotion does; a floating or complex input is read as it is. Without it rsqrt of an
  int64 2 truncated 0.7071 to 0 (Kokoro's vocoder residual collapsed to zeros in NeuroBrix, 2026).
"""
from __future__ import annotations

from typing import Optional, Sequence

from .itemsize import itemsize, kind

__all__ = ["promoted", "float_math_input"]


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
