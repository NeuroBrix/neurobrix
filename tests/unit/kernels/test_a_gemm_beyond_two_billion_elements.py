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
(the store faulted under launch blocking); GREEN after. Needs room for A and C at once
(5.5 GB + headroom, computed from M, N and K): skipped, and said, otherwise.

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src pytest tests/unit/kernels/test_a_gemm_beyond_two_billion_elements.py -p no:cacheprovider
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels import wrappers as W
from neurobrix.kernels.nbx_tensor import NBXTensor

M, N, K = 1_100_000, 2048, 64          # M*N = 2 252 800 000 > 2**31; rows >= 1 048 576 overflow an int32 offset
FIRST_OVERFLOWING_ROW = (2 ** 31) // N   # 1 048 576


def _device_free_bytes():
    """Free device memory, asked of the ENGINE rather than of CUDA.

    This was `ctypes.CDLL("libcudart.so")`, which does not exist off CUDA. On Apple the load
    raised, the helper returned 0, and `_require_room` skipped with "0.0 GB free" — a reason
    that reads as a busy card and is really "this probe cannot see this machine". So the gate
    for a defect that IS PRESENT on Metal could never run there (measured 2026-09-23: a matmul
    whose C holds 2 201 600 000 elements deviates 1.0 from the fp64 oracle on an M4 Pro, while
    2 048 000 000 elements deviates 0.002141 — the same bracket this file exists to hold).

    `DeviceAllocator.device_free_bytes` answers on every backend the engine supports, which is
    the right authority for a question about the engine's own device.
    """
    try:
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        free = DeviceAllocator.device_free_bytes(None)
        return int(free) if free and free > 0 else 0
    except Exception:                                    # noqa: BLE001
        return 0


#: A (M x K) and C (M x N), both fp16, live at once; B is negligible.
NEEDED_BYTES = (M * N + M * K) * 2 + (1 << 30)                 # + 1 GB headroom


def _require_room():
    """Refuse at entry, at the moment the memory is actually wanted.

    The former `@pytest.mark.skipif(_cuda_free_bytes() < 8 * 2 ** 30, ...)` ran as
    a DECORATOR ARGUMENT — at collection, against an empty card — and so could not
    see that by the time this test runs, earlier tests in the same process still
    hold their allocations. On 2026-09-18 it let the test start on a 16 GB card and
    the 4.5 GB output failed to allocate, which the merge gate reported as a
    kernel failure. The figure now comes from M, N and K above.
    """
    # `cudaMemGetInfo` reports what the DRIVER holds, and the NBX allocator's
    # free-list pool (on by default) keeps released blocks instead of returning
    # them. So a previous test's tensors read as "not free" while the allocator
    # could serve this test from them immediately. Measured 2026-09-18 on a 16 GB
    # card: after dropping two 4.2 GB tensors the driver reported 7.07 GB free with
    # the pool on and 15.47 GB with `NBX_ALLOC_POOL=0` — the same 8.4 GB, parked.
    # Flushing first makes the question answerable by the driver again, and
    # releasing the cache before a multi-gigabyte allocation is right anyway.
    try:
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        DeviceAllocator.empty_cache_pool()
    except Exception:
        pass
    free = _device_free_bytes()
    if free < NEEDED_BYTES:
        pytest.skip(
            f"needs {NEEDED_BYTES / 2 ** 30:.1f} GB free on the device — C is "
            f"{M * N * 2 / 2 ** 30:.1f} GB and A is {M * K * 2 / 2 ** 20:.0f} MB, "
            f"plus headroom; {free / 2 ** 30:.1f} GB free")


def test_rows_past_the_int32_boundary_hold_the_product():
    _require_room()
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
