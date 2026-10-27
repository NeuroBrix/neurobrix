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
  weight_load_gbps        the ENGINE'S OWN LOADER on components the caller names (`--weights <model>:<component>,...`,
                          about `--weights-gb` of each one's block weights): bytes landed / time, per engine (triton,
                          compiled), per STORED dtype (what the loader converts from), to the card ("device") and to
                          host memory ("host": the host-held path of zero3 and the expert stores), cold (the shards'
                          pages evicted first, posix_fadvise DONTNEED) and warm (the same load again). A load is read + decode + convert + land — the cost a plan
                          pays, which a bare copy rate (h2d_gbps) is not (the rule-13 proof, 2026-10-27: Wan2.2 lazy
                          loads 123-173 s, Qwen3-30B 294-447 s, against estimates at the H2D rate)

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


def system_share_mb(readings: int = 30, every_s: float = 10.0) -> dict:
    """The system's own share of the installed RAM: installed less the CALM available memory (median of `readings`
    taken `every_s` apart through `core.host_memory.memory_state`, the reading the planner and its door use). Taken on
    an idle machine — no run, no copy — it is what the OS and its resident applications hold whatever a run does;
    the unified pool is bounded by it (`PrismSolver._unified_host_pool_mb`)."""
    import statistics
    from neurobrix.core.host_memory import memory_state
    vals, total = [], None
    for i in range(int(readings)):
        st = memory_state()
        if not st.measured or not st.total_mb:
            raise RuntimeError(f"the host memory cannot be read here ({st.source}): no share is measured")
        vals.append(float(st.available_mb)); total = float(st.total_mb)
        if i + 1 < readings:
            time.sleep(every_s)
    calm = statistics.median(vals)
    return {"installed_mb": total, "calm_available_mb": calm, "min_mb": min(vals), "max_mb": max(vals),
            "system_share_mb": int(round(total - calm))}


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


def _evict(paths) -> None:
    for p in paths:
        fd = os.open(str(p), os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def block_keys(root: Path, comp: str, gb: float):
    """(keys, stored bytes, shard files) — the component's first block weights, in shard order, up to `gb` GB stored.
    Refuses a component with no shard or no block weight by name."""
    from neurobrix.triton.weight_loader import _read_header, is_block_key
    wdir = root / "components" / comp / "weights"
    shards = sorted(wdir.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"--weights {root.name}:{comp}: no shard under {wdir}")
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
        raise SystemExit(f"--weights {root.name}:{comp}: the component holds no block weight to load")
    return keys, stored, used


def stored_dtype(shards, keys) -> str:
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


def weight_load_gbps(spec: str, gb: float) -> dict:
    """{engine: {"device"|"host": {"cold": GB/s, "warm": GB/s}}} — the engine's own loader on `model:component` of
    the cache, its first block weights up to `gb` GB (stored), bytes LANDED (device or host) per second."""
    from neurobrix.core.paths import cache_dir
    model, comp = spec.split(":", 1)
    root = cache_dir() / model
    keys, stored, used = block_keys(root, comp, gb)
    dtype = stored_dtype(used, keys)
    only = set(keys)
    rel = {f"components/{comp}/weights/{sh.name}": "cpu" for sh in used}
    out = {"model": model, "component": comp, "stored_bytes": stored, "weights": len(keys), "dtype": dtype}

    def timed(load):
        t0 = time.perf_counter()
        landed = load()
        return landed / (time.perf_counter() - t0) / 1e9

    from neurobrix.kernels.nbx_tensor import NBXDtype, DeviceAllocator
    from neurobrix.triton.weight_loader import load_component_weights

    def triton(host):
        def load():
            w = load_component_weights(str(root), comp, 0, NBXDtype.float16, only=only,
                                       shard_map=rel if host else None)
            DeviceAllocator.sync_device()
            n = sum(t.nbytes() for k, t in w.items() if k != "_arenas")
            del w
            return n
        return load
    import torch
    from neurobrix.core.io import WeightLoader
    # the compiled engine's device as torch names it on this machine (a CUDA card, or the Apple GPU)
    tdev = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else None)
    if tdev is None:
        raise SystemExit("weight_load_gbps: no accelerator torch can see for the compiled engine's device load")

    def tsync():
        torch.cuda.synchronize() if tdev.startswith("cuda") else torch.mps.synchronize()

    def compiled(host):
        def load():
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
    for engine, make in (("triton", triton), ("compiled", compiled)):
        out[engine] = {}
        # one small load first, never timed: the engine's context, its conversion kernels' compilation and its pools
        # belong to the process, not to the load (the first triton bf16 load measured 0.01 GB/s cold without it)
        keep = only
        only = set(keys[:4])
        make(False)()
        make(True)()
        only = keep
        for where in ("device", "host"):
            _evict(used)
            cold = timed(make(where == "host"))
            warm = timed(make(where == "host"))
            out[engine][where] = {"cold": round(cold, 3), "warm": round(warm, 3)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--card", required=True, help="the card measured (as the profile indexes it)")
    ap.add_argument("--out", required=True, type=Path, help="where the facts are written (an atelier root)")
    ap.add_argument("--storage-file", type=Path, default=None, help="a file to read for the storage bandwidth")
    ap.add_argument("--weights", default=None,
                    help="model:component[,model:component...] of the cache whose block weights the engines' loaders "
                         "load (weight_load_gbps), one per stored dtype a plan meets (bf16, fp32, fp16)")
    ap.add_argument("--weights-gb", type=float, default=4.0, help="how many stored GB of them")
    ap.add_argument("--system-share", action="store_true",
                    help="also measure the system's share of the RAM (30 calm readings, 5 min; unified-memory hosts)")
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
    if a.system_share:
        facts["system_share"] = system_share_mb()
    if a.storage_file is not None:
        facts["storage_read_gbps"] = storage_read_gbps(a.storage_file)
        facts["storage_file"] = str(a.storage_file)
    if a.weights is not None:
        # {engine: {stored dtype: {device|host: {cold, warm}}}}, one measured component per stored dtype named
        runs = [weight_load_gbps(spec.strip(), a.weights_gb) for spec in a.weights.split(",") if spec.strip()]
        facts["weight_load_gbps"] = {e: {r["dtype"]: r[e] for r in runs} for e in ("triton", "compiled")}
        facts["weight_load_sources"] = [{k: r[k] for k in ("model", "component", "dtype", "stored_bytes", "weights")}
                                        for r in runs]
    (out / f"facts_card{a.card}.json").write_text(json.dumps(facts, indent=1))
    src = f"measured {facts['date']} on card {a.card} (tools/profile_facts.py)"
    print(json.dumps(facts, indent=1))
    print("\n# the hardware profile's block, each value with its source")
    _mm = ", ".join(f"{d}: {facts['matmul_tflops'][d]['best']:.1f}" for d in dtypes)
    print(f"measured:\n  source: \"{src}\"\n  matmul_tflops: {{{_mm}}}\n  memory_gbps: {facts['memory_gbps']:.0f}\n"
          f"  h2d_gbps: {facts['h2d_gbps']:.1f}\n  d2h_gbps: {facts['d2h_gbps']:.1f}\n"
          f"  h2d_pageable_gbps: {facts['h2d_pageable_gbps']:.1f}\n  host_matmul_gflops: {facts['host_matmul_gflops']:.0f}\n"
          f"  host_memory_gbps: {facts['host_memory_gbps']:.0f}")
    if "weight_load_gbps" in facts:
        wl = facts["weight_load_gbps"]
        srcs = ", ".join(f"{r['model']}:{r['component']} {r['dtype']} {r['stored_bytes'] / 1e9:.1f} GB"
                         for r in facts["weight_load_sources"])
        print(f"  weight_load_gbps:  # the engines' loaders, GB/s landed — {srcs}")
        for engine in ("triton", "compiled"):
            print(f"    {engine}:")
            for dt, ws in wl[engine].items():
                print(f"      {dt}: {{" + ", ".join(f"{w}: {{cold: {v['cold']:.2f}, warm: {v['warm']:.2f}}}"
                                             for w, v in ws.items()) + "}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
