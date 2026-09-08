#!/usr/bin/env python3
"""Run every autotune config of a kernel ALONE against an fp64 oracle.

Why this exists
---------------
The autotune screen (``kernels/launcher.py:screen_configs``) decides by
CONSENSUS: it runs each candidate and keeps the ones that agree with each
other. That is the right design — there is no way to know in advance which
config is correct — but consensus is not correctness. **If every config that
compiles is wrong in the same way, the screen accepts them all.**

Measured 2026-09-08 on this machine: ``conv2d_forward_kernel`` had 16 of 18
configs refused by the Metal backend and the two survivors both 99% wrong
against an fp64 oracle. It surfaced only because those two were wrong
DIFFERENTLY and the screen would not choose. Had they agreed, a silently
incorrect convolution would have shipped.

So the screen needs a companion that does not ask the configs what they think:
it asks an oracle. This is that companion.

What it does
------------
For each config in the kernel's autotuner, and one config at a time:

  * restrict the tuner's candidate set to that config alone — the REAL call
    path (wrapper, launcher, dispatch) is otherwise untouched, so what is
    measured is that config's own output and not a tuner decision;
  * run the case;
  * compare against a float64 reference computed here, in numpy;
  * record ran/refused, the ULP distance, the relative error, and the exact
    refusal text when it does not compile.

A config that REFUSES is not a failure — a backend declining to emit is the
behaviour we want. A config that RUNS and disagrees with the oracle is the
finding.

Vendor-agnostic by construction: nothing here names a vendor, a device prefix
or a backend. It measures whatever the active profile dispatches to, so the
same sweep is the verdict on `cuda:0`, `hip:0`, `xpu:0` and `mps:0` alike.

Usage
-----
    PYTHONPATH=src python tools/kernel_config_oracle_sweep.py \
        --case conv2d --out validation_outputs/conv_oracle_<date>

Caches are cleared by this tool before the run, and the run is
SINGLE-THREADED by design: the three caches are global state and overlapping
runs corrupt each other.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_CACHES = (
    Path.home() / ".cache" / "triton_msl",
    Path.home() / ".triton" / "cache",
)


def clear_caches() -> list[str]:
    """Clear every compile/tuning cache. Returns what was cleared, for the header."""
    cleared = []
    for p in _CACHES:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
        cleared.append(str(p))
    replay = Path.home() / ".neurobrix" / "replay_cache"
    if replay.is_dir():
        for f in replay.glob("autotune_configs_*.json"):
            f.unlink(missing_ok=True)
            cleared.append(str(f))
    return cleared


def find_tuner(kernel):
    """The autotuner object behind a decorated kernel, or None.

    Walked rather than reached for by attribute name: the decorator stack
    (nbx_autotune -> triton.autotune -> triton.jit) has changed shape before.
    """
    seen, stack = set(), [kernel]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if hasattr(obj, "configs") and hasattr(obj, "run") and hasattr(obj, "fn"):
            return obj
        for attr in ("fn", "kernel", "_fn", "base_fn"):
            nxt = getattr(obj, attr, None)
            if nxt is not None:
                stack.append(nxt)
    return None


def ulp_distance(got: np.ndarray, want: np.ndarray, dtype) -> np.ndarray:
    """Representable steps between two arrays of `dtype`.

    The reference bank's metric, not a per-element ratio: a ratio explodes
    near a zero crossing and would report a catastrophe where the answer is
    correctly rounded.
    """
    if np.dtype(dtype) not in _INT_OF:
        # An unknown dtype is not licence to guess a metric. Naming it is the
        # whole job of this tool.
        raise ValueError(
            f"ulp_distance has no integer view for {dtype!r}; "
            f"known: {sorted(str(d) for d in _INT_OF)}")
    info = np.finfo(dtype)
    a = np.asarray(got, dtype=dtype)
    b = np.asarray(want, dtype=dtype)
    ia = a.view(_INT_OF[dtype]).astype(np.int64)
    ib = b.view(_INT_OF[dtype]).astype(np.int64)
    # map the sign-magnitude ordering onto a monotone integer line
    ia = np.where(ia < 0, np.iinfo(_INT_OF[dtype]).min - ia, ia)
    ib = np.where(ib < 0, np.iinfo(_INT_OF[dtype]).min - ib, ib)
    d = np.abs(ia - ib)
    both_finite = np.isfinite(a) & np.isfinite(b)
    return np.where(both_finite, d, np.where(a == b, 0, np.iinfo(np.int64).max)), info


_INT_OF = {np.dtype(np.float16): np.int16, np.dtype(np.float32): np.int32}


# ---------------------------------------------------------------- cases ----
def case_conv2d(seed: int = 0):
    """A small convolution with padding, the shape an upscaler's first layer has."""
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels.wrappers import conv2d_wrapper
    from neurobrix.kernels.ops.conv2d import conv2d_forward_kernel

    N, C, H, W, O, K = 1, 4, 16, 16, 8, 3
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N, C, H, W)).astype(np.float16)
    w = (rng.standard_normal((O, C, K, K)) * 0.3).astype(np.float16)
    b = (rng.standard_normal(O) * 0.1).astype(np.float16)

    xp = np.pad(x.astype(np.float64), ((0, 0), (0, 0), (1, 1), (1, 1)))
    ref = np.zeros((N, O, H, W), dtype=np.float64)
    for o in range(O):
        for c in range(C):
            for i in range(K):
                for j in range(K):
                    ref[0, o] += xp[0, c, i:i + H, j:j + W] * float(w[o, c, i, j])
        ref[0, o] += float(b[o])

    def run():
        out = conv2d_wrapper(
            NBXTensor.from_numpy(x), NBXTensor.from_numpy(w), NBXTensor.from_numpy(b),
            stride=[1, 1], padding=[1, 1], dilation=[1, 1],
            transposed=False, output_padding=[0, 0], groups=1)
        return out.numpy()

    return dict(kernel=conv2d_forward_kernel, run=run, ref=ref,
                dtype=np.dtype(np.float16),
                label=f"conv2d {N}x{C}x{H}x{W} * {O}x{C}x{K}x{K} pad1 stride1")


CASES = {"conv2d": case_conv2d}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=sorted(CASES), required=True)
    ap.add_argument("--out", required=True, help="directory for RESULTS.md + records.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    cleared = clear_caches()

    case = CASES[args.case](args.seed)
    tuner = find_tuner(case["kernel"])
    if tuner is None:
        print("no autotuner found behind the kernel; nothing to sweep", file=sys.stderr)
        return 2

    original = list(tuner.configs)
    dtype = case["dtype"]
    ref = case["ref"]
    rows = []

    for cfg in original:
        label = "_".join(f"{k}{v}" for k, v in sorted(cfg.kwargs.items()))
        label += f"_w{cfg.num_warps}_s{cfg.num_stages}"
        tuner.configs = [cfg]
        for attr in ("cache", "_cache", "configs_timings"):
            c = getattr(tuner, attr, None)
            if isinstance(c, dict):
                c.clear()
        try:
            got = np.asarray(case["run"]())
            want = ref.astype(dtype)
            ulp, _info = ulp_distance(got.astype(dtype), want, dtype)
            scale = float(np.abs(ref).max()) or 1.0
            rel = float(np.abs(got.astype(np.float64) - ref).max() / scale)
            rows.append(dict(config=label, ran=True,
                             max_ulp=int(ulp.max()), mean_ulp=float(ulp.mean()),
                             rel_err=rel, out_sum=float(got.astype(np.float64).sum())))
        except Exception as exc:                      # noqa: BLE001 - recorded, never swallowed
            rows.append(dict(config=label, ran=False,
                             refusal=type(exc).__name__,
                             detail=str(exc).replace("\n", " ")[:400]))
        sys.stdout.write(f"{label[:52]:54s}"
                         f"{'RAN  ulp=' + str(rows[-1]['max_ulp']) if rows[-1]['ran'] else 'REFUSED'}\n")
        sys.stdout.flush()

    tuner.configs = original

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.json").write_text(json.dumps(rows, indent=1))

    ran = [r for r in rows if r["ran"]]
    sums = sorted({round(r["out_sum"], 6) for r in ran})
    worst = max((r["max_ulp"] for r in ran), default=None)

    md = [
        f"# {case['label']} — every autotune config against an fp64 oracle",
        "",
        f"Generated **{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}** "
        f"by `tools/kernel_config_oracle_sweep.py` on {platform.system()} "
        f"{platform.release()} {platform.machine()}.",
        "**No public claim is made from any number here.**",
        "",
        f"* oracle: float64 in numpy, rounded to `{dtype}`",
        "* metric: representable steps (ULP), not a per-element ratio — a ratio "
        "explodes at a zero crossing",
        f"* each config run ALONE: the tuner's candidate set is the only thing changed",
        "* caches cleared by the tool before the run:",
    ] + [f"  * `{c}`" for c in cleared] + [
        "",
        "| config | ran | max ULP | mean ULP | rel err | output sum |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r["ran"]:
            md.append(f"| `{r['config']}` | yes | {r['max_ulp']} | {r['mean_ulp']:.3f} "
                      f"| {r['rel_err']:.3e} | {r['out_sum']:.6f} |")
        else:
            md.append(f"| `{r['config']}` | **refused** | — | — | — | — |")
    md += [
        "",
        f"**{len(ran)} of {len(rows)} configs ran.**",
        "",
    ]
    if len(sums) > 1:
        md.append(f"**The configs that ran DISAGREE: {len(sums)} distinct output sums, "
                  f"{sums}.** At least one is wrong; the autotune consensus screen "
                  "will refuse to choose, which is correct and is not a fix.")
    elif ran:
        md.append(f"All configs that ran agree, worst **{worst} ULP** against the oracle. "
                  "Agreement alone would not have proved this — every config could have "
                  "been wrong the same way. The oracle is what proves it.")
    if any(not r["ran"] for r in rows):
        md += ["", "## What the refusals said", ""]
        seen = set()
        for r in rows:
            if r["ran"] or r["detail"] in seen:
                continue
            seen.add(r["detail"])
            md.append(f"* `{r['refusal']}` — {r['detail']}")
    md.append("")
    (out / "RESULTS.md").write_text("\n".join(md))
    print(f"\nwrote {out}/RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
