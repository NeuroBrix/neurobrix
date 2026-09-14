"""group_norm addresses every element of an input past 2^31 elements.

Mochi's VAE decoder, after the GEMM and the flat kernels were widened: the
op-by-op run died at `aten.native_group_norm::26` (2026-09-14 07:33) —
`chan_start * HW` and `batch_idx * C * HW` in int32, the tile form of the
same class (register 58). Every program id is now widened at its source.
Shape: N=1, C=64, HW=40 000 000 → 2 560 000 000 fp16 elements (5.1 GB in,
5.1 GB out); the group starting at channel 56 has base offset 2.24e9, past
2^31 (the first form used HW=35.2e6, whose last group started at 1.97e9 —
below the boundary, green on the old tree, a vacuous test caught by running
it where it had to fail). Channels 0, 55, 56 and 63 are read back.
Seen RED on the tree of 7ee3d7c, GREEN after. Skipped, and said, without
>= 10 GB free.
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels import wrappers as W
from neurobrix.kernels.nbx_tensor import NBXTensor

N, C, HW, GROUPS = 1, 64, 40_000_000, 8      # chan_start is a multiple of 8: 56 * HW = 2.24e9 > 2^31 (the first form, HW=35.2e6, never crossed it: 56 * 35.2e6 = 1.97e9 — a test green on the old tree, vacuous)


def _cuda_free_bytes():
    try:
        import ctypes
        cuda = ctypes.CDLL("libcudart.so")
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        return free.value if cuda.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) == 0 else 0
    except OSError:
        return 0


@pytest.mark.skipif(_cuda_free_bytes() < 10 * 2 ** 30, reason="needs a CUDA card with >= 10 GB free")
def test_channels_past_the_int32_boundary_are_normalised():
    x = NBXTensor.ones((N, C, HW), dtype="float16")
    y = W.group_norm_wrapper(x, GROUPS, None, None, 1e-5)
    y = y[0] if isinstance(y, tuple) else y          # the wrapper returns (y, mean, rstd)
    assert tuple(y.shape) == (N, C, HW)
    for c in (0, 55, 56, 63):
        row = y[0:1, c:c + 1, HW - 4:HW].contiguous().numpy().astype(np.float32).ravel()
        assert np.all(np.isfinite(row)), f"channel {c}: non-finite past the boundary"
        assert np.allclose(row, 0.0, atol=1e-2), f"channel {c}: a constant input normalises to 0, got {row}"
