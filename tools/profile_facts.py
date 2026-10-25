#!/usr/bin/env python3
"""The hardware facts Prism's time estimate reads, MEASURED on this machine (docs/design/prism-time-score.md §1).

One card at a time (`CUDA_VISIBLE_DEVICES`), through the engine's own primitives — the Triton matmul wrapper and its
timer (`launcher.do_bench`), the device allocator's copies — never a vendor library:

  matmul_tflops.<dtype>   a square GEMM at the sizes below, 2*M*N*K / time, the best of the sizes
  memory_gbps             a device-to-device copy, (read + write) bytes / time
  h2d_gbps / d2h_gbps     a copy from page-locked host memory to the card and back
  h2d_pageable_gbps       the same from ordinary host memory (what a loader without staging pays)
  host_matmul_gflops      a float32 GEMM on the host through numpy's BLAS, the machine's thread setting
  host_memory_gbps        a copy of a 1 GiB array in host memory, (read + write) bytes / time
  storage_read_gbps       a sequential read of a file the caller names (a weight shard of the cache)

Writes `<out>/facts_card<N>.json` and prints the YAML block to copy into the hardware profile, each value with its
source line. The autotuner's replay cache is redirected into `<out>` (`NEUROBRIX_REPLAY_CACHE`): a size the directory
does not certify is swept there and never kept by the machine.

    CUDA_VISIBLE_DEVICES=2 profile_facts.py --card 2 --out <campaign>/facts --storage-file <a shard>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atelier import writable  # noqa: E402

GEMM_SIZES = (2048, 4096, 8192)
COPY_BYTES = 1 << 30


def _sync():
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    DeviceAllocator.device_synchronize()


def matmul_tflops(dtype: str) -> dict:
    """The GEMM throughput at `dtype`. numpy draws float16 and float32; bfloat16 (no numpy dtype) is drawn float32
    and cast on the device by the engine's own cast (`NBXTensor.to`), so the measured kernel is the bf16 matmul a
    bf16 plan launches — the time estimate reads `matmul_tflops.<compute dtype>` and refuses by name without it
    (the Mac, 2026-10-25 18:10: no estimate on 88 of 98 Metal plans for want of this figure)."""
    import numpy as np
    from neurobrix.kernels import launcher as L
    from neurobrix.kernels import wrappers as W
    from neurobrix.kernels.nbx_tensor import NBXTensor
    npd = {"float16": np.float16, "float32": np.float32, "bfloat16": np.float32}[dtype]
    rng = np.random.default_rng(0)
    per = {}
    for n in GEMM_SIZES:
        a = NBXTensor.from_numpy(np.ascontiguousarray((rng.standard_normal((n, n)) * 0.1).astype(npd))).to_cuda(0).to(dtype)
        b = NBXTensor.from_numpy(np.ascontiguousarray((rng.standard_normal((n, n)) * 0.1).astype(npd))).to_cuda(0).to(dtype)
        ms = float(L.do_bench(lambda: W.mm(a, b), warmup=50, rep=200))
        per[n] = 2.0 * n * n * n / (ms * 1e-3) / 1e12
        del a, b
    return {"best": max(per.values()), "per_size": per}


def _timed_copies(dst: int, src: int, nbytes: int, kind: int, reps: int) -> float:
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    DeviceAllocator.memcpy(dst, src, nbytes, kind)
    _sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        DeviceAllocator.memcpy(dst, src, nbytes, kind)
    _sync()
    return (time.perf_counter() - t0) / reps


def copies() -> dict:
    import numpy as np
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    n = COPY_BYTES
    d0, d1 = DeviceAllocator.malloc_cuda(n), DeviceAllocator.malloc_cuda(n)
    h = DeviceAllocator.malloc_host_pinned(n)
    pageable = np.ones(n, dtype=np.uint8)
    try:
        t_dd = _timed_copies(d1, d0, n, 3, 20)
        t_hd = _timed_copies(d0, h, n, 1, 10)
        t_dh = _timed_copies(h, d0, n, 2, 10)
        t_pd = _timed_copies(d0, pageable.ctypes.data, n, 1, 5)
    finally:
        DeviceAllocator.free_cuda(d0)
        DeviceAllocator.free_cuda(d1)
        DeviceAllocator.free_host_pinned(h)
    g = lambda b, t: b / t / 1e9                                                    # noqa: E731
    return {"memory_gbps": g(2 * n, t_dd), "h2d_gbps": g(n, t_hd), "d2h_gbps": g(n, t_dh),
            "h2d_pageable_gbps": g(n, t_pd)}


def host_matmul_gflops() -> float:
    import numpy as np
    n = 4096
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, n), dtype=np.float32)
    b = rng.standard_normal((n, n), dtype=np.float32)
    a @ b
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        a @ b
        best = min(best, time.perf_counter() - t0)
    return 2.0 * n ** 3 / best / 1e9


def host_memory_gbps() -> float:
    import numpy as np
    src = np.ones(COPY_BYTES, dtype=np.uint8)
    dst = np.empty_like(src)
    np.copyto(dst, src)
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        np.copyto(dst, src)
        best = min(best, time.perf_counter() - t0)
    return 2.0 * COPY_BYTES / best / 1e9


def storage_read_gbps(path: Path) -> float:
    size = path.stat().st_size
    fd = os.open(str(path), os.O_RDONLY)
    try:
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)        # cold: not the page cache
        t0 = time.perf_counter()
        while os.read(fd, 64 << 20):
            pass
        return size / (time.perf_counter() - t0) / 1e9
    finally:
        os.close(fd)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--card", required=True, help="the card measured (as the profile indexes it)")
    ap.add_argument("--out", required=True, type=Path, help="where the facts are written (an atelier root)")
    ap.add_argument("--storage-file", type=Path, default=None, help="a file to read for the storage bandwidth")
    ap.add_argument("--dtypes", default="float16,float32",
                    help="the GEMM dtypes measured; add bfloat16 on hardware that computes it natively (the profile's "
                         "has_native_bf16: Apple M-series, Ampere and later) — the plan's compute dtype must be among them")
    a = ap.parse_args(argv)
    dtypes = tuple(d.strip() for d in a.dtypes.split(",") if d.strip())
    out = writable(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["NEUROBRIX_REPLAY_CACHE"] = str(out / "replay_cache")
    facts = {"card": a.card, "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
             "matmul_tflops": {d: matmul_tflops(d) for d in dtypes}, **copies(),
             "host_matmul_gflops": host_matmul_gflops(), "host_memory_gbps": host_memory_gbps()}
    if a.storage_file is not None:
        facts["storage_read_gbps"] = storage_read_gbps(a.storage_file)
        facts["storage_file"] = str(a.storage_file)
    (out / f"facts_card{a.card}.json").write_text(json.dumps(facts, indent=1))
    src = f"measured {facts['date']} on card {a.card} (tools/profile_facts.py)"
    print(json.dumps(facts, indent=1))
    print("\n# the hardware profile's block, each value with its source")
    _mm = ", ".join(f"{d}: {facts['matmul_tflops'][d]['best']:.1f}" for d in dtypes)
    print(f"measured:\n  source: \"{src}\"\n  matmul_tflops: {{{_mm}}}\n  memory_gbps: {facts['memory_gbps']:.0f}\n"
          f"  h2d_gbps: {facts['h2d_gbps']:.1f}\n  d2h_gbps: {facts['d2h_gbps']:.1f}\n"
          f"  h2d_pageable_gbps: {facts['h2d_pageable_gbps']:.1f}\n  host_matmul_gflops: {facts['host_matmul_gflops']:.0f}\n"
          f"  host_memory_gbps: {facts['host_memory_gbps']:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
