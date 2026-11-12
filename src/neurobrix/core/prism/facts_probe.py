"""The hardware facts Prism's time estimate reads, MEASURED (the device's speeds by nbxtensor's backend.device_measure) — ONE code for the
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
  vmm_chunk_bytes         the expandable pool's chunk (R16): the device's allocation granularity x k, k the smallest whose
                          map + unmap costs no more than an ordinary allocation of the same bytes (its median plus its
                          spread) — the supervisor's criterion of 2026-11-02 06:40; absent when the device has no VMM
  expert_copy             the expert store's miss copy, measured by the library next to its kernel
                          (`nbxtensor.ops.moe_device_residency.measure_miss_copy`, autodetect's first launch)
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

# The device's speeds are measured by the library (its kernels, its copies); the page cache is evicted by it too.
from nbxtensor.backend.device_measure import copies, matmul_tflops  # noqa: E402,F401  (read here as P.copies, ...)
from nbxtensor.io.read_rate import cold_read_gbps as storage_read_gbps, evict  # noqa: E402,F401


def gemv_gbps(dtype: str, engine: str, k: int, n: int, reps: int) -> float:
    """The weight bandwidth one decode step's matrix-vector product reaches on the ENGINE: x[1, k] @ W[n, k]^T, the
    linear of a generated token, timed over `reps` launches; W's bytes / time. A decode is weight-bound and an engine
    reaches a fraction of the device's copy bandwidth on it (the Mac, 2026-10-27 23:02: Triton on the M4 Pro ~1/15 —
    Janus-Pro-7B estimated 41-47 s at the device bandwidth, measured 445 s)."""
    if engine == "triton":
        from nbxtensor.backend.device_measure import gemv_gbps as _library_gemv
        return _library_gemv(dtype, k, n, reps)
    import numpy as np
    rng = np.random.default_rng(0)
    npd = {"float16": np.float16, "float32": np.float32, "bfloat16": np.float32}[dtype]
    xs = np.ascontiguousarray((rng.standard_normal((1, k)) * 0.1).astype(npd))
    ws = np.ascontiguousarray((rng.standard_normal((n, k)) * 0.05).astype(npd))
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


def block_keys(root: Path, comp: str, gb: float) -> Tuple[List[str], int, List[Path]]:
    """(keys, stored bytes, shard files) — the component's first block weights, in shard order, up to `gb` GB stored.
    Refuses a component with no shard or no block weight by name."""
    from neurobrix.triton.weight_loader import is_block_key, read_safetensors_header as _read_header
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
    from neurobrix.triton.weight_loader import read_safetensors_header as _read_header
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


def weight_load_gbps(root: Path, comp: str, gb: float) -> dict:
    """{"dtype": stored dtype, "stored_bytes", "weights", engine: {"device"|"host": {"cold": GB/s, "warm": GB/s}}} —
    the engine's own loader on the component's first block weights up to `gb` GB stored, bytes LANDED per second."""
    root = Path(root)
    keys, stored, used = block_keys(root, comp, gb)
    out = {"dtype": stored_dtype(used, keys), "stored_bytes": stored, "weights": len(keys)}

    def timed(load) -> float:
        t0 = time.perf_counter()
        landed = load()
        return round(landed / (time.perf_counter() - t0) / 1e9, 3)

    for engine in ("triton", "compiled"):        # both engines' loaders
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


def vmm_first_replay(device: int):
    """(replay s, child s) of the chunk probe's first replay, made ahead; None where the device has no VMM."""
    from nbxtensor.memory import trace_replay as T
    return None if T.vmm_granularity(device) is None else T.first_replay(device)


def vmm_chunk_bytes(device: int, reps: int, ks: Sequence[int], off_first: Optional[float] = None) -> Optional[int]:
    """The expandable pool's chunk on `device`, measured on a REAL run's allocations (R16; the supervisor 2026-11-02
    17:15): the library's recorded allocation trace (`nbxtensor.memory.trace_replay`) replayed through the allocator on
    this device, under the card size it was recorded on, with the range off (the driver segments) and then at each
    chunk = granularity x k, `ks` ascending, `reps` replays each in its own process; the chunk is the first whose median
    replay is within R16 OFF's median plus its spread. None when the device has no VMM, or when no chunk keeps up with
    the segments (the pool keeps them). On the V100 16 GB: OFF 2.16-2.29 s, 16 MiB 2.7 s, 32 MiB 2.0 s -> 32 MiB; the
    run itself, SANA-Video_2B_720p ttft 40.6/41.4 s at 32 MiB against 40.9/39.9 s OFF (vmm_v100.md). A synthetic
    churn of the same bytes was measured and rejected: its cost per call is flat, so it always favours the largest."""
    from nbxtensor.memory import trace_replay as T
    gran = T.vmm_granularity(device)
    return None if gran is None else T.chunk_by_replay(device, gran, ks, reps, off_first=off_first)[0]


def card_facts_in_a_child(index: int) -> dict:
    """`autodetect.runtime_card_facts(index)` taken in its OWN short process: reading what the runtime takes on a card
    means initialising it, and a launch holds a context only on the cards its plan names — the launching process must
    not keep one on every card of the profile for having completed it. {} when the child cannot answer."""
    import json
    import subprocess
    import sys
    code = ("import json; from neurobrix.core.prism import autodetect as A; "
            f"print('CARD_FACTS ' + json.dumps(A.runtime_card_facts({int(index)})))")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, env=env)
    said = [line[len("CARD_FACTS "):] for line in out.stdout.splitlines() if line.startswith("CARD_FACTS ")]
    return json.loads(said[-1]) if out.returncode == 0 and said else {}


def complete_runtime_card_facts(devices, measure) -> dict:
    """The runtime's total and its own context on each card of a GENERATED profile that states neither
    (written at generation since 2026-10-27): measured now by `measure(index)` — `card_facts_in_a_child`: autodetect's own reading in its own process, {}
    when the runtime cannot answer — and set on the device entry. A
    profile generated before, or at a moment the runtime could not answer, stayed without them for ever and every
    plan on it was refused by name, telling the user to delete the file (the two-card `default-*` profile of the rack,
    2026-11-11). Returns {"runtime_card_facts": [device indices completed]} or {}."""
    done = []
    for device in devices:
        if device.get("unified_memory") or device.get("runtime_context_mb") is not None:
            continue
        facts = measure(int(device.get("index", 0)))
        if facts:
            device.update(facts)
            done.append(int(device.get("index", 0)))
    return {"runtime_card_facts": done} if done else {}
