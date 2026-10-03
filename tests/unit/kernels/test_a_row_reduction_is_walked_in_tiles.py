"""A row reduction holds at most one tile of its row: sum, mean and amax walk a long row in tiles.

2026-10-03, Wan2.1-I2V at 480x832 on a 16 GB card: the transformer's first kernels raised the
driver's used memory by 4 356 MB while the allocator tracked 72 MB, and the run died at SDPA::0
short of 836 MB. The op-level probe named it: `sum_wrapper` over [0,1,3,4] of a [1,36,12,60,104]
tensor (a 224 640-element row) sized its tile to the whole row — 2 048 fp32 values per thread at
4 warps — which spill to local memory, and the driver reserves local memory for every thread the
card can hold: +2 062 MB for that one sum, never given back.

What would this file do if the code were wrong? The tile back at `next_power_of_2(feat_dim)` ->
the long-row launch reserves gigabytes, the first test RED; the loop dropping a tile or reading
past the row -> the values against float64, the second RED. Skipped, and said, without a card.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]


def _cuda_free_bytes():
    try:
        import ctypes
        cuda = ctypes.CDLL("libcudart.so")
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        return free.value if cuda.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) == 0 else 0
    except OSError:
        return 0


needs_card = pytest.mark.skipif(_cuda_free_bytes() < 2 * 2 ** 30, reason="needs a CUDA card with >= 2 GB free")


@needs_card
def test_a_long_row_reduction_reserves_no_local_memory():
    """In a fresh process (the driver's reservation never shrinks, so nothing may launch before):
    the driver's used memory around one sum over a 449 280-element row (Wan2.1-I2V's at CFG 2)."""
    code = textwrap.dedent("""
        import ctypes
        from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator, _gpu_runtime
        from neurobrix.kernels import wrappers as W
        rt = _gpu_runtime()
        def used():
            f, t = ctypes.c_size_t(), ctypes.c_size_t(); rt.cudaMemGetInfo(ctypes.byref(f), ctypes.byref(t))
            return (t.value - f.value) >> 20
        x = NBXTensor.empty((2, 36, 12, 60, 104), NBXDtype.float32, "cuda:0"); DeviceAllocator.sync_device()
        u0 = used(); W.sum_wrapper(x, dim=[0, 1, 3, 4]); DeviceAllocator.sync_device()
        print(used() - u0)
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600,
                       env={"PYTHONPATH": str(REPO / "src"), "PYTHONNOUSERSITE": "1", "PATH": "/usr/bin:/bin",
                            **{k: v for k, v in __import__("os").environ.items() if k.startswith(("CUDA", "LD_", "HOME"))}})
    assert r.returncode == 0, r.stderr[-2000:]
    grew = int(r.stdout.strip().splitlines()[-1])
    assert grew < 256, f"one row sum grew the driver's used memory by {grew} MB (local memory of a spilling tile)"


@needs_card
@pytest.mark.parametrize("feat", [1, 63, 64, 4095, 4096, 4097, 8192, 12289, 224640])
def test_sum_mean_amax_agree_with_float64_at_every_row_length(feat):
    from neurobrix.kernels import wrappers as W
    from neurobrix.kernels.nbx_tensor import NBXTensor
    rng = np.random.default_rng(feat)
    a = rng.standard_normal((3, feat)).astype(np.float32)
    a[1, -1] = 50.0                                         # the row's maximum in its LAST element
    x = NBXTensor.from_numpy(a)
    ref = a.astype(np.float64)
    tol = 1e-5 * np.sqrt(feat) + 1e-5
    np.testing.assert_allclose(W.sum_wrapper(x, dim=1).numpy(), ref.sum(1), rtol=1e-5, atol=tol * 10)
    np.testing.assert_allclose(W.mean_wrapper(x, dim=1).numpy(), ref.mean(1), rtol=1e-5, atol=tol)
    np.testing.assert_array_equal(W.amax_wrapper(x, dim=1).numpy(), a.max(1))
