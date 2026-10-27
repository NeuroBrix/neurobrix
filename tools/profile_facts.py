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

# The measuring primitives are the engine's own (core/prism/facts_probe.py): one code for this tool's full measurement
# and for a machine's first launch (the supervisor's S2, 2026-10-27).
from neurobrix.core.prism.facts_probe import (  # noqa: E402
    block_keys, copies, host_matmul_gflops, host_memory_gbps, matmul_tflops, storage_read_gbps, stored_dtype)
from neurobrix.core.prism import facts_probe as _probe  # noqa: E402


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


def weight_load_gbps(spec: str, gb: float) -> dict:
    """The engines' loaders on `model:component` of the cache (`facts_probe.weight_load_gbps`), named."""
    from neurobrix.core.paths import cache_dir
    model, comp = spec.split(":", 1)
    try:
        r = _probe.weight_load_gbps(cache_dir() / model, comp, gb)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    return {"model": model, "component": comp, **r}


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
