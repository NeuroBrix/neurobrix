"""The attention family's float64 reference: softmax(q k^T * scale + mask) v, on query-row windows.

A reference must not share an implementation with the kernels it checks: numpy in float64, no Triton, no torch. K and
V are widened ONCE and the window's two products go through BLAS (`np.matmul`, the host's threads): widened again in
every window by a one-core `np.einsum`, a key over 262 144 cached positions held its card idle for minutes while the
host computed its reference (2026-11-12, moved here from the engine's certifier the same day with that fix: 8.4 s ->
0.8 s on a 65 536-key sample, the values equal to 5e-16).

The caller decides the windows (how many rows a reference may cost) and how an operand's numbers are read (`widen`:
a bf16 operand's bits decoded, for instance); this module computes.
"""
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


def attention_fp64(q, k, v, scale: float, windows: Sequence[Tuple[int, int]], mask=None,
                   widen: Callable = np.asarray) -> List[Tuple[Tuple[int, int], np.ndarray]]:
    """[((r0, r1), the reference's rows r0..r1 [..., r1 - r0, Dv] in float64)] for q [..., Tq, D], k [..., Tk, D],
    v [..., Tk, Dv], an additive mask [..., Tq, Tk] or None."""
    kt = np.swapaxes(widen(k).astype(np.float64), -1, -2)
    vf = widen(v).astype(np.float64)
    out = []
    for r0, r1 in windows:
        s = np.matmul(widen(q[..., r0:r1, :]).astype(np.float64), kt) * scale
        if mask is not None:
            s = s + widen(mask[..., r0:r1, :]).astype(np.float64)
        s = s - s.max(axis=-1, keepdims=True)
        e = np.exp(s)
        out.append(((r0, r1), np.matmul(e / e.sum(axis=-1, keepdims=True), vf)))
    return out
