#!/usr/bin/env python3
"""Measure cumsum against an fp64 oracle across every chunk boundary.

A scan wider than the 1024-thread threadgroup is lowered by decomposing the
tile into one CONTIGUOUS range per thread. The defect class that decomposition
introduces is an off-by-one in the last partial range, so this sweeps the
lengths where a range boundary falls — T-1, T, T+1, 2T, 2T+1, and the widths
the reference bank uses — in both narrow dtypes.

ULP is the reference bank's own metric (representable steps between the result
and the fp64 oracle rounded to the result's dtype). A per-element error ratio
is NOT used: a cumsum of signed values crosses zero, and near a crossing that
ratio explodes while saying nothing about the result's quality.

    python tools/cumsum_width_sweep.py --out validation_outputs/<dated>/
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_CACHES = (Path.home() / ".cache" / "triton_msl",
           Path.home() / ".triton" / "cache")


def clear_caches() -> list:
    cleared = []
    for path in _CACHES:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        cleared.append(str(path))
    replay = Path.home() / ".neurobrix" / "replay_cache"
    if replay.exists():
        for f in replay.glob("autotune_configs_*.json"):
            f.unlink()
            cleared.append(str(f))
    return cleared


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rows", type=int, default=3)
    args = ap.parse_args()

    cleared = clear_caches()

    sys.path.insert(0, "src")
    spec = importlib.util.spec_from_file_location("bank", "tools/metal_bank_replay.py")
    bank = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bank)
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels.wrappers import cumsum_wrapper

    lengths = [1, 2, 511, 1023, 1024, 1025, 1026, 2047, 2048, 2049,
               3077, 4095, 4096, 4097, 5000, 8192, 8193]
    rows = []
    for dt in (np.float16, np.float32):
        for n in lengths:
            rng = np.random.default_rng(n)
            x = rng.standard_normal((args.rows, n)).astype(dt)
            rec = {"dtype": dt.__name__, "n": n}
            # the client's own tiling, so the row says which path it measured
            blk = min(int(2 ** np.ceil(np.log2(max(n, 1)))), 4096)
            parts = -(-n // blk)
            rec["block_size"] = blk
            rec["parts"] = parts
            rec["path"] = ("one thread per element" if blk <= 1024
                           else "chunked (one contiguous range per thread)")
            try:
                got = cumsum_wrapper(NBXTensor.from_numpy(x), 1).numpy()
            except Exception as e:  # a refusal is a measurement, not a gap
                rec["status"] = "refused"
                rec["reason"] = f"{type(e).__name__}: {e}"
                rows.append(rec)
                continue
            oracle = np.cumsum(x.astype(np.float64), axis=1)
            rec["status"] = "ok"
            rec.update(bank.ulp_distance(got, oracle))
            rows.append(rec)

    args.out.mkdir(parents=True, exist_ok=True)
    doc = {
        "generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "tool": "tools/cumsum_width_sweep.py",
        "machine": f"{platform.system()} {platform.release()} {platform.machine()}",
        "oracle": "numpy.cumsum in float64, rounded to the result's dtype",
        "metric": "reference-bank ulp_distance (representable steps)",
        "caches_cleared_before_the_run": cleared,
        "rows_per_case": args.rows,
        "results": rows,
    }
    (args.out / "records.json").write_text(json.dumps(doc, indent=2))

    md = [f"# cumsum against an fp64 oracle, across every chunk boundary", "",
          f"Generated **{doc['generated']}** by `{doc['tool']}` on {doc['machine']}.",
          "**No public claim is made from any number here.**", "",
          f"* oracle: {doc['oracle']}",
          f"* metric: {doc['metric']}",
          f"* {args.rows} rows per case, seeded per length",
          "* caches cleared by the tool before the run:"]
    md += [f"  * `{c}`" for c in cleared]
    md += ["", "| dtype | n | tile | parts | path | max ULP | mean ULP | max abs err |",
           "|---|---:|---:|---:|---|---:|---:|---:|"]
    for r in rows:
        if r["status"] == "refused":
            md.append(f"| {r['dtype']} | {r['n']} | {r['block_size']} | {r['parts']} | "
                      f"{r['path']} | refused | — | {r['reason'][:60]} |")
        else:
            md.append(f"| {r['dtype']} | {r['n']} | {r['block_size']} | {r['parts']} | "
                      f"{r['path']} | {r['max_ulp']} | {r['mean_ulp']:.3f} | "
                      f"{r['max_abs_err']:.3e} |")
    (args.out / "RESULTS.md").write_text("\n".join(md) + "\n")
    over = [r for r in rows if r.get("max_ulp", 0) > 1]
    print(f"{len(rows)} cases -> {args.out}/RESULTS.md ({len(over)} over 1 ULP)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
