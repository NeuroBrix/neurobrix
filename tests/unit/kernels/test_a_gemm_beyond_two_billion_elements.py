"""A GEMM whose output holds more than 2^31 elements addresses every element.

`mochi-1-preview`'s VAE (`aten.mm::1`, M=1 068 480 × N=2048 at its runtime
latent size, 33 264 at the trace) died on CUDA error 700 at every attempt
since 2026-09-10 (D-MOCHI-CUDA-700-AT-MM); two `compute-sanitizer` runs of
7 200 s and 18 000 s measured nothing; `--triton-sequential` with
`CUDA_LAUNCH_BLOCKING=1` named the op in 19 minutes (2026-09-14 05:34). The
kernel computed `stride_cm * offs_cm` with `offs_cm` an int32 arange: past
row 2^31 / N the offset wraps negative and the store goes to an address the
allocation does not own. A trace at M=33 264 can never see it — the shape is
part of the test, and this one is chosen so M*N exceeds 2^31 by a margin
while the tensors fit a 32 GB card (C fp16 = 4.5 GB, A = 141 MB):
M = 1 100 000, N = 2048, K = 64 → 2 252 800 000 elements; the first row
overflowing is 1 048 576.

Seen RED on 2026-09-14 05:37 UTC before the int64 promotion of the row/column offsets
(the store faulted under launch blocking); GREEN after. Needs a CUDA card with
>= 8 GB free: skipped, and said, otherwise.

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src pytest tests/unit/kernels/test_a_gemm_beyond_two_billion_elements.py -p no:cacheprovider
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels import wrappers as W
from neurobrix.kernels.nbx_tensor import NBXTensor

M, N, K = 1_100_000, 2048, 64          # M*N = 2 252 800 000 > 2**31; rows >= 1 048 576 overflow an int32 offset
FIRST_OVERFLOWING_ROW = (2 ** 31) // N   # 1 048 576


def _cuda_free_bytes():
    try:
        import ctypes
        cuda = ctypes.CDLL("libcudart.so")
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        if cuda.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) != 0:
            return 0
        return free.value
    except OSError:
        return 0


@pytest.mark.skipif(_cuda_free_bytes() < 8 * 2 ** 30, reason="needs a CUDA card with >= 8 GB free (C is 4.5 GB)")
def test_rows_past_the_int32_boundary_hold_the_product():
    rng = np.random.default_rng(20260914)
    a = rng.standard_normal((M, K), dtype=np.float32).astype(np.float16)
    b = rng.standard_normal((K, N), dtype=np.float32).astype(np.float16)
    c = W.mm(NBXTensor.from_numpy(a), NBXTensor.from_numpy(b))
    assert tuple(c.shape) == (M, N)
    rows = [0, FIRST_OVERFLOWING_ROW - 1, FIRST_OVERFLOWING_ROW, FIRST_OVERFLOWING_ROW + 1, M - 1]
    ref = a[rows].astype(np.float32) @ b.astype(np.float32)
    for i, r in enumerate(rows):
        got = c[r:r + 1].contiguous().numpy().astype(np.float32)[0]
        assert np.allclose(got, ref[i], rtol=2e-2, atol=2e-1), (
            f"row {r} ({'past' if r >= FIRST_OVERFLOWING_ROW else 'before'} the int32 boundary at {FIRST_OVERFLOWING_ROW}): "
            f"max |diff| {np.abs(got - ref[i]).max():.3g} — the store's offset arithmetic does not address this row")
