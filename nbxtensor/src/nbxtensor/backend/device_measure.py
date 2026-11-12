"""A device's speeds, measured here through the library's own kernels and copies: what an application's planner reads
to estimate a run's time (NeuroBrix's Prism records them in a hardware profile).

- `matmul_tflops(dtype, sizes, warmup, rep)`: a square GEMM through `linalg_wrappers.mm`, 2*n^3 / time, per size and
  the best of them. bfloat16 (no numpy dtype) is drawn float32 and cast on the device, so the kernel timed is the bf16
  matmul a bf16 run launches.
- `gemv_gbps(dtype, k, n, reps)`: one decode step's matrix-vector product, x[1, k] @ W[n, k]^T, W's bytes / time.
- `copies(nbytes, reps)`: device to device ((read + write) bytes / time), page-locked host to device and back, and
  ordinary (pageable) host memory to device.

The operands are drawn from `numpy.random.default_rng(0)`, so two machines time the same values. Torch-free.
"""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np

__all__ = ["matmul_tflops", "gemv_gbps", "copies"]

_DRAWN = {"float16": np.float16, "float32": np.float32, "bfloat16": np.float32}


def _on_device(a: np.ndarray, dtype: str):
    from nbxtensor.tensor import NBXTensor
    return NBXTensor.from_numpy(np.ascontiguousarray(a.astype(_DRAWN[dtype]))).to_cuda(0).to(dtype)


def matmul_tflops(dtype: str, sizes: Sequence[int] = (2048, 4096, 8192), warmup: int = 50, rep: int = 200) -> dict:
    """{"best": TFLOP/s, "per_size": {n: TFLOP/s}} of a square GEMM at `dtype` on device 0."""
    from nbxtensor.backend import launcher as L
    from nbxtensor.ops import linalg_wrappers as LW
    rng = np.random.default_rng(0)
    per = {}
    for n in sizes:
        a = _on_device(rng.standard_normal((n, n)) * 0.1, dtype)
        b = _on_device(rng.standard_normal((n, n)) * 0.1, dtype)
        ms = float(L.do_bench(lambda: LW.mm(a, b), warmup=warmup, rep=rep))
        per[n] = 2.0 * n * n * n / (ms * 1e-3) / 1e12
        del a, b
    return {"best": max(per.values()), "per_size": per}


def gemv_gbps(dtype: str, k: int, n: int, reps: int) -> float:
    """GB/s of weights one decode step's x[1, k] @ W[n, k]^T streams on device 0, timed over `reps` launches."""
    from nbxtensor.backend import launcher as L
    from nbxtensor.ops import linalg_wrappers as LW
    rng = np.random.default_rng(0)
    x = _on_device(rng.standard_normal((1, k)) * 0.1, dtype)
    w = _on_device(rng.standard_normal((n, k)) * 0.05, dtype)
    wt = w.t()
    ms = float(L.do_bench(lambda: LW.mm(x, wt), warmup=10, rep=reps))
    return round(w.nbytes() / (ms * 1e-3) / 1e9, 1)


def _timed_copies(dst: int, src: int, nbytes: int, kind: int, reps: int) -> float:
    from nbxtensor.memory.allocator import DeviceAllocator as DA
    DA.memcpy(dst, src, nbytes, kind)
    DA.device_synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        DA.memcpy(dst, src, nbytes, kind)
    DA.device_synchronize()
    return (time.perf_counter() - t0) / reps


def copies(nbytes: int = 1 << 30, reps: int = 10) -> dict:
    """{"memory_gbps", "h2d_gbps", "d2h_gbps", "h2d_pageable_gbps"} for copies of `nbytes` on the current device."""
    from nbxtensor.memory.allocator import DeviceAllocator as DA
    n = int(nbytes)
    d0, d1 = DA.malloc_cuda(n), DA.malloc_cuda(n)
    h = DA.malloc_host_pinned(n)
    pageable = np.ones(n, dtype=np.uint8)
    try:
        t_dd = _timed_copies(d1, d0, n, 3, 2 * reps)
        t_hd = _timed_copies(d0, h, n, 1, reps)
        t_dh = _timed_copies(h, d0, n, 2, reps)
        t_pd = _timed_copies(d0, pageable.ctypes.data, n, 1, max(1, reps // 2))
    finally:
        DA.free_cuda(d0)
        DA.free_cuda(d1)
        DA.free_host_pinned(h)
    return {"memory_gbps": 2 * n / t_dd / 1e9, "h2d_gbps": n / t_hd / 1e9, "d2h_gbps": n / t_dh / 1e9,
            "h2d_pageable_gbps": n / t_pd / 1e9}
