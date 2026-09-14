"""Flat-indexed kernels address every element of a tensor past 2^31 elements.

The GEMM epilogue was the first site (register 58, Mochi's VAE); the same
int32 offset form sat in 129 flat-indexed kernels — fill, the strided copy
that materialises a view, the elementwise family. This exercises three of
them on one 2 252 800 000-element fp16 tensor (4.5 GB; a 32 GB card):
`fill` writes it, a transposed view is materialised by the strided copy, and
`add` reads and writes it; the elements before, at and past 2^31 are read
back. Seen RED before the promotion (2026-09-14: the strided copy of a
transposed 2.2e9-element view faulted / wrote garbage past 2^31), GREEN
after. Skipped, and said, without >= 10 GB free.

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src pytest tests/unit/kernels/test_a_flat_kernel_beyond_two_billion_elements.py -p no:cacheprovider
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels import wrappers as W
from neurobrix.kernels.nbx_tensor import NBXTensor

ROWS, COLS = 1_100_000, 2048            # 2 252 800 000 elements; the int32 boundary falls inside row 1 048 576
BOUNDARY = 2 ** 31


def _cuda_free_bytes():
    try:
        import ctypes
        cuda = ctypes.CDLL("libcudart.so")
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        return free.value if cuda.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) == 0 else 0
    except OSError:
        return 0


def _at(t, flat_index):
    r, c = divmod(flat_index, COLS)
    return float(t[r:r + 1].contiguous().numpy()[0, c])


@pytest.mark.skipif(_cuda_free_bytes() < 10 * 2 ** 30, reason="needs a CUDA card with >= 10 GB free")
def test_fill_strided_copy_and_add_reach_past_the_boundary():
    probes = [0, BOUNDARY - 1, BOUNDARY, BOUNDARY + 1, ROWS * COLS - 1]
    x = NBXTensor.ones((ROWS, COLS), dtype="float16")           # the fill kernel writes 2.25e9 elements
    for p in probes:
        assert _at(x, p) == 1.0, f"fill did not reach element {p}"
    y = x.t().contiguous()                 # (COLS, ROWS) materialised by the strided copy
    assert tuple(y.shape) == (COLS, ROWS)
    last = y[COLS - 1:COLS].contiguous().numpy()[0]
    assert last.shape == (ROWS,) and float(last[-1]) == 1.0 and float(last[0]) == 1.0, "the strided copy lost the far end"
    z = W.add(x, x)
    for p in probes:
        assert _at(z, p) == 2.0, f"add did not reach element {p}"
