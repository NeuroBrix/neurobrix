#!/usr/bin/env python
"""Measure what a bucketed autotune key costs against the per-size optimum.

The owner's decision (2026-09-21): request-dependent dimensions are bucketed in the
launcher's key — the bucket selects the configuration, the kernel runs the true size with
its masks. The ladder is chosen by measurement, the way the tile unit was: for one kernel
and one fixed shape, sweep the request-dependent dimension over the sizes a request can
produce; at every size the autotuner benches EVERY viable configuration (the sweep the
engine already runs on a miss, with its consensus screen), so one sweep per size yields
the per-size optimum AND the time of any other configuration at that size. A ladder is
then evaluated offline: each size is served the configuration certified for its bucket's
representative (the bucket's TOP — proven at a size at least as large as any it serves),
and the loss is time(bucket config at size) / time(optimum at size) - 1.

    CUDA_VISIBLE_DEVICES=0 NBX_AUTOTUNE_CERTIFIED=off NEUROBRIX_REPLAY_CACHE=/tmp/x \\
      python tools/bucket_loss.py --kernel matmul --dim M --fixed N=2048,K=2048 \\
      --sizes 1-64,65,80,96,112,128,129,160,192,224,256,257,320,384,448,512,513,640,768,896,1024,1025,1280,1536,1792,2048,2049,3072,4096 \\
      --out nbx/campaigns/2026_09_21_bucketed_keys/matmul_M_2048x2048_card0.json
    python tools/bucket_loss.py --evaluate nbx/campaigns/.../matmul_M_2048x2048_card0.json \\
      --ladder L16 --ladder Lpow2 --ladder Lmix

Kernels: matmul (mm, dim M), baddbmm (bmm scores, dims M or N), conv2d (dims H/W, B).
The directory is OFF and the replay cache is private so every size sweeps. A sweep is
the autotuner's own bench (triton do_bench on this card, warm); the figures are this
card's and go to owed-proofs with the memory class named.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def parse_sizes(spec: str):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


# --------------------------------------------------------------------------- ladders
def ladder_L16(v):
    if v <= 256:
        return -(-v // 16) * 16
    if v <= 1024:
        return -(-v // 32) * 32
    if v <= 8192:
        return -(-v // 128) * 128
    return -(-v // 512) * 512


def ladder_Lpow2(v):
    if v <= 64:
        p = 1
        while p < v:
            p *= 2
        return p
    if v <= 512:
        return -(-v // 64) * 64
    p = 512
    while p < v:
        p *= 2
    return p


def ladder_Lmix(v):
    return v if v <= 64 else ladder_L16(v)


LADDERS = {"L16": ladder_L16, "Lpow2": ladder_Lpow2, "Lmix": ladder_Lmix, "exact": lambda v: v}


# --------------------------------------------------------------------------- one sweep
def _cfg_repr(cfg) -> str:
    return json.dumps({"kwargs": dict(cfg.kwargs), "num_warps": cfg.num_warps,
                       "num_stages": cfg.num_stages}, sort_keys=True)


def sweep_matmul(M, N, K, dtype, dev):
    import numpy as np
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels import wrappers as W
    from neurobrix.kernels.ops.matmul import matmul_kernel
    from neurobrix.triton import autotune_cache as atc
    rng = np.random.default_rng(M)
    a = NBXTensor.from_numpy((rng.standard_normal((M, K)) * 0.1).astype(dtype))   # lands on the visible card
    b = NBXTensor.from_numpy((rng.standard_normal((K, N)) * 0.1).astype(dtype))
    seen = {}
    saved = matmul_kernel.run

    def spy(*args, **kwargs):
        seen["key"] = atc.key_of(matmul_kernel, args, kwargs)
        return saved(*args, **kwargs)
    matmul_kernel.run = spy
    try:
        matmul_kernel.cache.clear()
        W.mm(a, b)
    finally:
        matmul_kernel.run = saved
    key = seen.get("key")
    best = matmul_kernel.cache.get(key)
    timings = getattr(matmul_kernel, "configs_timings", None) or {}
    row = {"size": M, "key": atc.key_repr(key) if hasattr(atc, "key_repr") else repr(key),
           "best": _cfg_repr(best) if best else None,
           "timings_ms": {_cfg_repr(c): (float(t[0]) if isinstance(t, (list, tuple)) else float(t))
                          for c, t in timings.items() if t is not None}}
    return row


def sweep_bmm(B, M, N, K, dtype):
    """The batched GEMM of the SDPA math path: scores = Q @ K^T at [B, M, K] x [B, K, N]. The
    replay census (302 baddbmm keys on this machine) has M > 1 in 301 of them — the prefill
    scores at the request's length — with N the key length and K the head dim."""
    import numpy as np
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels import wrappers as W
    from neurobrix.kernels.ops.baddbmm_op import baddbmm_kernel
    from neurobrix.triton import autotune_cache as atc
    rng = np.random.default_rng(M * 1000 + N)
    a = NBXTensor.from_numpy((rng.standard_normal((B, M, K)) * 0.1).astype(dtype))
    b = NBXTensor.from_numpy((rng.standard_normal((B, K, N)) * 0.1).astype(dtype))
    seen = {}
    saved = baddbmm_kernel.run

    def spy(*args, **kwargs):
        seen["key"] = atc.key_of(baddbmm_kernel, args, kwargs)
        return saved(*args, **kwargs)
    baddbmm_kernel.run = spy
    try:
        baddbmm_kernel.cache.clear()
        W.bmm(a, b)
    finally:
        baddbmm_kernel.run = saved
    key = seen.get("key")
    best = baddbmm_kernel.cache.get(key)
    timings = getattr(baddbmm_kernel, "configs_timings", None) or {}
    return {"size": M if N is None else (M, N), "key": atc.key_repr(key) if hasattr(atc, "key_repr") else repr(key),
            "best": _cfg_repr(best) if best else None,
            "timings_ms": {_cfg_repr(c): (float(t[0]) if isinstance(t, (list, tuple)) else float(t))
                           for c, t in timings.items() if t is not None}}


def measure(a):
    # The wrappers' dtype policy exactly as before a request (IEEE precision, operand
    # promotion, the store dtype): without it the tool sweeps a kernel variant no model
    # meets — the first run keyed (M, N, K, False, False, fp16, fp16, fp16) where
    # TinyLlama's live keys read (M, N, K, True, True, fp16, fp16, fp32).
    from neurobrix.kernels.autotune_certify import _bind_hardware_profile
    print(f"[bucket_loss] hardware profile bound: {_bind_hardware_profile()}", flush=True)
    fixed = dict(kv.split("=") for kv in a.fixed.split(",")) if a.fixed else {}
    fixed = {k: int(v) for k, v in fixed.items()}
    sizes = parse_sizes(a.sizes)
    dev = int(a.device)
    rows = []
    t0 = time.time()
    for s in sizes:
        t1 = time.time()
        if a.kernel == "matmul":
            row = sweep_matmul(s if a.dim == "M" else fixed["M"], fixed.get("N", s), fixed["K"], a.dtype, dev)
        elif a.kernel == "bmm":
            B = fixed.get("B", 32)
            if a.dim == "M":            # prefill scores: M = N = the request's length, K = head dim
                row = sweep_bmm(B, s, fixed.get("N", s) if "N" in fixed else s, fixed["K"], a.dtype)
            else:                       # the key length grows, the query length fixed
                row = sweep_bmm(B, fixed["M"], s, fixed["K"], a.dtype)
            row["size"] = s
        else:
            raise SystemExit(f"kernel {a.kernel!r}: not wired (matmul, bmm)")
        row["wall_s"] = round(time.time() - t1, 2)
        rows.append(row)
        n_cfg = len(row["timings_ms"])
        best_ms = min(row["timings_ms"].values()) if row["timings_ms"] else None
        print(f"[bucket_loss] {a.kernel} {a.dim}={s:5d} configs={n_cfg:3d} best={best_ms} ms  ({row['wall_s']} s)", flush=True)
    doc = {"kernel": a.kernel, "dim": a.dim, "fixed": fixed, "dtype": a.dtype, "device": dev,
           "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "rows": rows,
           "wall_s": round(time.time() - t0, 1)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print(f"[bucket_loss] written {a.out} ({len(rows)} sizes, {doc['wall_s']} s)")


def evaluate(a):
    doc = json.loads(Path(a.evaluate).read_text())
    rows = {r["size"]: r for r in doc["rows"] if r.get("timings_ms")}
    sizes = sorted(rows)
    print(f"{doc['kernel']} dim {doc['dim']} fixed {doc['fixed']} on device {doc['device']}: {len(sizes)} sizes")
    for name in a.ladder:
        fn = LADDERS[name]
        losses = []
        per_bucket = {}
        missing = 0
        for s in sizes:
            top = fn(s)
            # the representative measured for this bucket: the largest measured size <= top
            reps = [x for x in sizes if x <= top and fn(x) == top]
            rep = max(reps) if reps else None
            if rep is None:
                missing += 1
                continue
            cfg = rows[rep]["best"]
            t_opt = min(rows[s]["timings_ms"].values())
            t_bucket = rows[s]["timings_ms"].get(cfg)
            if t_bucket is None:
                missing += 1
                continue
            loss = t_bucket / t_opt - 1.0
            losses.append(loss)
            per_bucket.setdefault(top, []).append(loss)
        if not losses:
            print(f"  {name}: no size evaluable"); continue
        losses_sorted = sorted(losses)
        med = losses_sorted[len(losses_sorted) // 2]
        worst = max(losses)
        n_buckets = len(per_bucket)
        print(f"  {name:6s}: buckets={n_buckets:3d} sizes={len(losses):3d} median loss={med * 100:5.1f} %  "
              f"max loss={worst * 100:5.1f} %  unevaluable={missing}")
        if a.verbose:
            for top in sorted(per_bucket):
                ls = per_bucket[top]
                print(f"      bucket top {top:5d}: n={len(ls):2d} median={sorted(ls)[len(ls) // 2] * 100:5.1f} % max={max(ls) * 100:5.1f} %")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="matmul")
    ap.add_argument("--dim", default="M")
    ap.add_argument("--fixed", default="N=2048,K=2048")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--sizes", default="1-64")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--evaluate", default=None)
    ap.add_argument("--ladder", action="append", default=[])
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.evaluate:
        return evaluate(a)
    if not a.out:
        raise SystemExit("--out is required for a measurement")
    if os.environ.get("NBX_AUTOTUNE_CERTIFIED") != "off":
        raise SystemExit("refused: run with NBX_AUTOTUNE_CERTIFIED=off so every size sweeps")
    if not os.environ.get("NEUROBRIX_REPLAY_CACHE"):
        raise SystemExit("refused: run with a private NEUROBRIX_REPLAY_CACHE so the machine's cache is untouched")
    measure(a)


if __name__ == "__main__":
    sys.exit(main() or 0)
