"""The hardware facts Prism's time estimate reads, MEASURED through the engine's own primitives — ONE code for the
measuring tool (`tools/profile_facts.py`, the full measurement a committed profile records) and for the first launch on
a machine (`autodetect`, `ensure_loader_rates`: the quick measurement a generated profile records), so a machine whose
profile and vendor file carry no measured facts is measured where it runs instead of planned without them (the
supervisor's S2, 2026-10-27 22:20).

  matmul_tflops.<dtype>   a square GEMM through the Triton matmul wrapper, 2*M*N*K / time, the best of the sizes
  memory_gbps             a device-to-device copy, (read + write) bytes / time
  h2d_gbps / d2h_gbps     a copy from page-locked host memory to the card and back
  h2d_pageable_gbps       the same from ordinary host memory
  host_matmul_gflops      a float32 GEMM on the host through numpy's BLAS
  host_memory_gbps        a copy of an array in host memory, (read + write) bytes / time
  storage_read_gbps       a sequential read of a named file, its pages dropped first
  gemv_gbps               a decode step's matrix-vector product on each engine, the weight bytes it streams / time
  weight_load_gbps        the ENGINE'S OWN LOADER on a component's first block weights: bytes landed / time, per engine,
                          per stored dtype, to the card ("device") and to host memory ("host": the host-held path),
                          cold (the shards' pages dropped first) and warm (the same load again)

R33: torch is imported by the compiled engine's loader measurement only; a Triton process never imports it here.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _sync() -> None:
    from nbxtensor.memory.allocator import DeviceAllocator
    DeviceAllocator.device_synchronize()


def matmul_tflops(dtype: str, sizes: Sequence[int] = (2048, 4096, 8192), warmup: int = 50, rep: int = 200) -> dict:
    """The GEMM throughput at `dtype`. numpy draws float16 and float32; bfloat16 (no numpy dtype) is drawn float32 and
    cast on the device by the engine's own cast, so the measured kernel is the bf16 matmul a bf16 plan launches."""
    import numpy as np
    from nbxtensor.backend import launcher as L
    from nbxtensor.ops import wrappers as W
    from nbxtensor.ops import linalg_wrappers as _nbx_linalg_wrappers  # the family the moved names live in (M2 split)
    from nbxtensor.tensor import NBXTensor
    npd = {"float16": np.float16, "float32": np.float32, "bfloat16": np.float32}[dtype]
    rng = np.random.default_rng(0)
    per = {}
    for n in sizes:
        a = NBXTensor.from_numpy(np.ascontiguousarray((rng.standard_normal((n, n)) * 0.1).astype(npd))).to_cuda(0).to(dtype)
        b = NBXTensor.from_numpy(np.ascontiguousarray((rng.standard_normal((n, n)) * 0.1).astype(npd))).to_cuda(0).to(dtype)
        ms = float(L.do_bench(lambda: _nbx_linalg_wrappers.mm(a, b), warmup=warmup, rep=rep))
        per[n] = 2.0 * n * n * n / (ms * 1e-3) / 1e12
        del a, b
    return {"best": max(per.values()), "per_size": per}


def gemv_gbps(dtype: str, engine: str, k: int = 4096, n: int = 14336, reps: int = 50) -> float:
    """The weight bandwidth one decode step's matrix-vector product reaches on the ENGINE: x[1, k] @ W[n, k]^T, the
    linear of a generated token, timed over `reps` launches; W's bytes / time. A decode is weight-bound and an engine
    reaches a fraction of the device's copy bandwidth on it (the Mac, 2026-10-27 23:02: Triton on the M4 Pro ~1/15 —
    Janus-Pro-7B estimated 41-47 s at the device bandwidth, measured 445 s)."""
    import numpy as np
    rng = np.random.default_rng(0)
    npd = {"float16": np.float16, "float32": np.float32, "bfloat16": np.float32}[dtype]
    xs = np.ascontiguousarray((rng.standard_normal((1, k)) * 0.1).astype(npd))
    ws = np.ascontiguousarray((rng.standard_normal((n, k)) * 0.05).astype(npd))
    if engine == "triton":
        from nbxtensor.backend import launcher as L
        from nbxtensor.ops import wrappers as W
        from nbxtensor.ops import linalg_wrappers as _nbx_linalg_wrappers  # the family the moved names live in (M2 split)
        from nbxtensor.tensor import NBXTensor
        x = NBXTensor.from_numpy(xs).to_cuda(0).to(dtype)
        w = NBXTensor.from_numpy(ws).to_cuda(0).to(dtype)
        wt = w.t()
        ms = float(L.do_bench(lambda: _nbx_linalg_wrappers.mm(x, wt), warmup=10, rep=reps))
        nbytes = w.nbytes()
    else:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "mps"
        td = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
        x = torch.from_numpy(xs).to(dev).to(td)
        w = torch.from_numpy(ws).to(dev).to(td)
        sync = torch.cuda.synchronize if dev == "cuda" else torch.mps.synchronize
        for _ in range(10):
            torch.mm(x, w.t())
        sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            torch.mm(x, w.t())
        sync()
        ms = (time.perf_counter() - t0) / reps * 1e3
        nbytes = w.numel() * w.element_size()
    return round(nbytes / (ms * 1e-3) / 1e9, 1)


def _timed_copies(dst: int, src: int, nbytes: int, kind: int, reps: int) -> float:
    from nbxtensor.memory.allocator import DeviceAllocator
    DeviceAllocator.memcpy(dst, src, nbytes, kind)
    _sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        DeviceAllocator.memcpy(dst, src, nbytes, kind)
    _sync()
    return (time.perf_counter() - t0) / reps


def copies(nbytes: int = 1 << 30, reps: int = 10) -> dict:
    import numpy as np
    from nbxtensor.memory.allocator import DeviceAllocator
    n = int(nbytes)
    d0, d1 = DeviceAllocator.malloc_cuda(n), DeviceAllocator.malloc_cuda(n)
    h = DeviceAllocator.malloc_host_pinned(n)
    pageable = np.ones(n, dtype=np.uint8)
    try:
        t_dd = _timed_copies(d1, d0, n, 3, 2 * reps)
        t_hd = _timed_copies(d0, h, n, 1, reps)
        t_dh = _timed_copies(h, d0, n, 2, reps)
        t_pd = _timed_copies(d0, pageable.ctypes.data, n, 1, max(1, reps // 2))
    finally:
        DeviceAllocator.free_cuda(d0)
        DeviceAllocator.free_cuda(d1)
        DeviceAllocator.free_host_pinned(h)
    g = lambda b, t: b / t / 1e9                                                    # noqa: E731
    return {"memory_gbps": g(2 * n, t_dd), "h2d_gbps": g(n, t_hd), "d2h_gbps": g(n, t_dh),
            "h2d_pageable_gbps": g(n, t_pd)}


def host_matmul_gflops(n: int = 4096) -> float:
    import numpy as np
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


def host_memory_gbps(nbytes: int = 1 << 30) -> float:
    import numpy as np
    src = np.ones(int(nbytes), dtype=np.uint8)
    dst = np.empty_like(src)
    np.copyto(dst, src)
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        np.copyto(dst, src)
        best = min(best, time.perf_counter() - t0)
    return 2.0 * int(nbytes) / best / 1e9


def drop_cache(fd: int, size: int) -> None:
    """Drop a file's pages from the OS page cache: posix_fadvise(DONTNEED) where it exists (Linux); on macOS, which has
    no posix_fadvise, msync(MS_INVALIDATE) over a shared read-only mapping of the file (the Mac, eb15e505, measured on
    the M4 Pro 2026-10-27: a shard read at 21 GB/s warm reads at 6.5 GB/s, the SSD's own rate, right after)."""
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return
    import ctypes
    import mmap as _mmap
    libc = ctypes.CDLL(None)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    if size <= 0:
        return
    addr = libc.mmap(None, size, _mmap.PROT_READ, _mmap.MAP_SHARED, fd, 0)
    if addr in (None, ctypes.c_void_p(-1).value):
        raise OSError(f"mmap of fd {fd} failed: cannot drop its pages from the cache")
    try:
        if libc.msync(addr, size, 2):          # MS_INVALIDATE
            raise OSError(f"msync(MS_INVALIDATE) failed on fd {fd}")
    finally:
        libc.munmap(addr, size)


def evict(paths: Iterable[Path]) -> None:
    for p in paths:
        fd = os.open(str(p), os.O_RDONLY)
        try:
            drop_cache(fd, os.fstat(fd).st_size)
        finally:
            os.close(fd)


def storage_read_gbps(path: Path) -> float:
    size = path.stat().st_size
    fd = os.open(str(path), os.O_RDONLY)
    try:
        drop_cache(fd, size)                   # cold: not the page cache
        t0 = time.perf_counter()
        while os.read(fd, 64 << 20):
            pass
        return size / (time.perf_counter() - t0) / 1e9
    finally:
        os.close(fd)


def block_keys(root: Path, comp: str, gb: float) -> Tuple[List[str], int, List[Path]]:
    """(keys, stored bytes, shard files) — the component's first block weights, in shard order, up to `gb` GB stored.
    Refuses a component with no shard or no block weight by name."""
    from neurobrix.triton.weight_loader import _read_header, is_block_key
    wdir = Path(root) / "components" / comp / "weights"
    shards = sorted(wdir.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"weight_load_gbps {Path(root).name}:{comp}: no shard under {wdir}")
    keys, stored, used = [], 0, []
    for sh in shards:
        header, _ = _read_header(str(sh))
        for k, info in header.items():
            if k == "__metadata__" or not is_block_key(k):
                continue
            a, b = info["data_offsets"]
            keys.append(k)
            stored += b - a
            if sh not in used:
                used.append(sh)
            if stored >= gb * 1e9:
                break
        if stored >= gb * 1e9:
            break
    if not keys:
        raise RuntimeError(f"weight_load_gbps {Path(root).name}:{comp}: the component holds no block weight to load")
    return keys, stored, used


def stored_dtype(shards: Iterable[Path], keys: Iterable[str]) -> str:
    """The safetensors dtype holding most of `keys`' stored bytes ("BF16", "F32", "F16") — a loader's rate depends on
    what it converts from (bf16 widened on the card, fp32 narrowed by numpy, fp16 as is)."""
    from neurobrix.triton.weight_loader import _read_header
    want, by = set(keys), {}
    for sh in shards:
        h, _ = _read_header(str(sh))
        for k, v in h.items():
            if k in want:
                a, b = v["data_offsets"]
                by[v["dtype"]] = by.get(v["dtype"], 0) + b - a
    return max(by, key=by.get)


def _triton_loader(root: Path, comp: str, used: List[Path]):
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.memory.allocator import DeviceAllocator
    from neurobrix.triton.weight_loader import load_component_weights
    rel = {f"components/{comp}/weights/{sh.name}": "cpu" for sh in used}

    def make(host: bool, only: set):
        def load() -> int:
            w = load_component_weights(str(root), comp, 0, NBXDtype.float16, only=only,
                                       shard_map=rel if host else None)
            DeviceAllocator.sync_device()
            n = sum(t.nbytes() for k, t in w.items() if k != "_arenas")
            del w
            return n
        return load
    return make


def _compiled_loader(root: Path, comp: str, used: List[Path]):
    import torch
    from neurobrix.core.io import WeightLoader
    # the compiled engine's device as torch names it on this machine (a CUDA card, or the Apple GPU)
    tdev = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else None)
    if tdev is None:
        raise RuntimeError("weight_load_gbps: no accelerator torch can see for the compiled engine's device load")

    def tsync():
        from neurobrix.core.device_utils import device_sync
        device_sync(tdev)

    def make(host: bool, only: set):
        def load() -> int:
            with WeightLoader(str(root)) as ld:
                w = ld.load_component(comp, "cpu" if host else tdev, torch.float16, only=only)
            if host:
                # the compiled engine's host-held path pins what it loads (core/strategies/zero3.py,
                # lazy_sequential.py: `tensor.contiguous().pin_memory()`) — a mapped view not yet read is no load
                w = {k: t.contiguous().pin_memory() for k, t in w.items()}
            tsync()
            n = sum(t.numel() * t.element_size() for t in w.values())
            del w
            return n
        return load
    return make


def weight_load_gbps(root: Path, comp: str, gb: float, engines: Sequence[str] = ("triton", "compiled")) -> dict:
    """{"dtype": stored dtype, "stored_bytes", "weights", engine: {"device"|"host": {"cold": GB/s, "warm": GB/s}}} —
    the engine's own loader on the component's first block weights up to `gb` GB stored, bytes LANDED per second."""
    root = Path(root)
    keys, stored, used = block_keys(root, comp, gb)
    out = {"dtype": stored_dtype(used, keys), "stored_bytes": stored, "weights": len(keys)}

    def timed(load) -> float:
        t0 = time.perf_counter()
        landed = load()
        return round(landed / (time.perf_counter() - t0) / 1e9, 3)

    for engine in engines:
        make = {"triton": _triton_loader, "compiled": _compiled_loader}[engine](root, comp, used)
        # one small load first, never timed: the engine's context, its conversion kernels' compilation and its pools
        # belong to the process, not to the load (the first triton bf16 load measured 0.01 GB/s cold without it)
        make(False, set(keys[:4]))()
        make(True, set(keys[:4]))()
        out[engine] = {}
        for where in ("device", "host"):
            evict(used)
            cold = timed(make(where == "host", set(keys)))
            warm = timed(make(where == "host", set(keys)))
            out[engine][where] = {"cold": cold, "warm": warm}
    return out
