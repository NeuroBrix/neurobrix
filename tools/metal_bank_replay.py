#!/usr/bin/env python3
"""Replay the Dell's kernel reference bank on this machine.

Each `.npz` in the bank holds the seeded inputs an op was run with, the
output CUDA produced, and an fp64 oracle. This runs the same op here through
the engine, and reports the Metal distance to that oracle beside the CUDA
distance already recorded in the file's `meta`.

    python tools/metal_bank_replay.py --bank <dir> --out results.json
    python tools/metal_bank_replay.py --bank <dir> --only matmul,addmm

The bar the owner set: **Metal ULP no greater than CUDA ULP**, per kernel and
per shape. A kernel that refuses is reported as refused with its reason — it
is a measured gap, not a missing row.

The bank lives on a read-only mount; nothing here writes to it.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np


def ulp_distance(got: np.ndarray, oracle: np.ndarray) -> dict:
    """Distance in ULP of the result's own dtype, plus the plain errors.

    The oracle is fp64; it is rounded to the result's dtype first, so the
    comparison is "how many representable steps away", which is what the
    bank's own numbers mean.
    """
    dtype = got.dtype
    rounded = oracle.astype(dtype)
    # The sign-magnitude -> ordered mapping is done in int64 throughout: the
    # bias for fp16 is 0x8000, which does not fit in the int16 the bits are
    # VIEWED as, and constructing it there raises rather than wrapping.
    view = {np.dtype(np.float16): np.int16,
            np.dtype(np.float32): np.int32}.get(dtype, np.int64)
    bias = {np.int16: 0x8000, np.int32: 0x80000000}.get(
        view, 0x8000000000000000)

    def ordered(v):
        i = v.view(view).astype(np.int64)
        return np.where(i < 0, np.int64(bias) - i, i)

    finite = np.isfinite(got) & np.isfinite(rounded)
    ulp = np.abs(ordered(got[finite]) - ordered(rounded[finite])) if finite.any() \
        else np.array([0])
    scale = float(np.abs(oracle).max()) or 1.0
    return {
        "max_ulp": int(ulp.max()),
        "mean_ulp": float(ulp.mean()),
        "max_abs_err": float(np.abs(got.astype(np.float64) - oracle).max()),
        "rel_err": float(np.abs(got.astype(np.float64) - oracle).max() / scale),
        "nonfinite": int((~np.isfinite(got)).sum()),
        "identical": bool(np.array_equal(got, rounded)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--only", default=None,
                        help="comma-separated op substrings")
    args = parser.parse_args()

    from neurobrix.kernels import launcher
    launcher.install()

    wanted = [w.strip() for w in args.only.split(",")] if args.only else None
    rows = []
    files = sorted(p for p in args.bank.rglob("*.npz"))
    for path in files:
        payload = np.load(path, allow_pickle=True)
        meta = json.loads(str(payload["meta"]))
        op = meta["op"]
        if wanted and not any(w in op for w in wanted):
            continue
        row = {"op": op, "tag": meta["tag"], "file": path.name,
               "launched": meta.get("launched", []),
               "cuda": meta.get("stats", [])}
        started = time.time()
        try:
            got = run_op(op, payload, meta)
            row["metal"] = [ulp_distance(g, payload[f"oracle{i}"])
                            for i, g in enumerate(got)]
            row["status"] = "ok"
        except Exception as exc:
            row["status"] = "refused"
            row["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:220]}"
            row["traceback_tail"] = traceback.format_exc().splitlines()[-1][:200]
        row["wall_s"] = round(time.time() - started, 4)
        rows.append(row)
        mark = "ok " if row["status"] == "ok" else "REF"
        extra = ""
        if row["status"] == "ok" and row["cuda"] and row["metal"]:
            extra = (f"metal_ulp={row['metal'][0]['max_ulp']:<8}"
                     f"cuda_ulp={row['cuda'][0].get('max_ulp')}")
        print(f"  {mark} {op:<38} {meta['tag']:<22} {extra}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1))
    refused = [r for r in rows if r["status"] != "ok"]
    print(f"\n{len(rows)} entries, {len(refused)} refused -> {args.out}")
    return 0


def run_op(op: str, payload, meta):
    """Run one bank entry through the engine's own wrappers."""
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels import wrappers

    inputs = [NBXTensor.from_numpy(np.ascontiguousarray(payload[k]))
              for k in sorted(payload.files) if k.startswith("in")]
    kwargs = dict(meta.get("kwargs") or {})

    handler = _HANDLERS.get(op)
    if handler is None:
        raise NotImplementedError(f"no bank handler for {op!r}")
    out = handler(wrappers, inputs, kwargs)
    outs = out if isinstance(out, (tuple, list)) else [out]
    return [np.asarray(o.numpy()) for o in outs]


_HANDLERS = {
    "scaled_dot_product_attention":
        lambda w, i, k: w.scaled_dot_product_attention_wrapper(*i[:3], **k),
    "_scaled_dot_product_efficient_attention":
        lambda w, i, k: w.scaled_dot_product_attention_wrapper(*i[:3], **k),
    "mm": lambda w, i, k: w.mm(i[0], i[1]),
    "matmul": lambda w, i, k: w.mm(i[0], i[1]),
    "addmm": lambda w, i, k: w.addmm(i[0], i[1], i[2]),
    "bmm": lambda w, i, k: w.bmm(i[0], i[1]),
    "baddbmm": lambda w, i, k: w.baddbmm_wrapper(i[0], i[1], i[2]),
}


if __name__ == "__main__":
    raise SystemExit(main())
