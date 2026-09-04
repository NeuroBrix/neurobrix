#!/usr/bin/env python3
"""Warm-request GPU timeline of the diffusers competitor arm (same row
recipe as diffusers_cell.py: fp16, stock SDPA, checkpoint scheduler).

Runs inside ~/bench_venvs/diffusers. One warm-up call, then the second
call wrapped in cudaProfilerStart/Stop for
`nsys profile --capture-range=cudaProfilerApi`. Reads the row from
benchmarks/config/rows.yml so both arms profile the same task.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diffusers_cell import load_pipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True)
    ap.add_argument("--rows-yml", default=str(Path(__file__).resolve().parents[1] / "config" / "rows.yml"))
    args = ap.parse_args()
    import yaml
    rows = {r["id"]: r for r in yaml.safe_load(open(args.rows_yml))["rows"]}
    row = rows[args.row]
    import torch
    pipe, cold, fixes, src, cache_note = load_pipeline(row)
    print(f"[warm] load {cold:.2f}s fixes={fixes}", flush=True)
    kw = {"prompt": row["prompt"], "num_inference_steps": row["steps"],
          "height": row.get("height"), "width": row.get("width")}
    kw = {k: v for k, v in kw.items() if v is not None}

    def one():
        gen = torch.Generator(device="cuda").manual_seed(row["seed"])
        with torch.inference_mode():
            return pipe(generator=gen, **kw)

    t0 = time.perf_counter(); one(); torch.cuda.synchronize()
    print(f"[warm] warm-up {time.perf_counter()-t0:.3f}s", flush=True)
    torch.cuda.cudart().cudaProfilerStart()
    t0 = time.perf_counter(); out = one(); torch.cuda.synchronize(); wall = time.perf_counter() - t0
    torch.cuda.cudart().cudaProfilerStop()
    out.images[0].save("/tmp/diffusers_warm_profiled.png")
    print(f"[warm] PROFILED request {wall:.3f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
