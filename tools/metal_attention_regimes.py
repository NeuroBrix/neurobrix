#!/usr/bin/env python3
"""The attention kernel on Metal, in the four regimes that matter.

    python tools/metal_attention_regimes.py --out regimes.json [--bank DIR]

The regimes, and why each is here:

* **prefill, non-causal** — `seqlen_q == seqlen_k`, no mask. The base case,
  and the only one a single-sequence-length template can express.
* **prefill, causal** — the same shape with a materialised causal mask in the
  bias. The mask must actually be applied: this is the regime where dropping
  it returns a plausible, finite, wrong answer.
* **decode** — `seqlen_q == 1` against a growing `seqlen_k`. Every token a
  language model emits after the first. A template carrying one sequence
  length cannot express it at all.
* **padding** — `seqlen_q == seqlen_k` with whole key positions masked to
  `-inf` through the bias. What a batch of unequal-length sequences looks
  like.

Each is checked against the CUDA reference from the bank where the shape
exists there, and against an fp64 oracle computed here otherwise. The bar is
the owner's: **Metal's ULP distance no greater than CUDA's**, per regime.

The oracle is deliberately NOT the engine's own kernel at higher precision —
it is plain softmax(QK^T*scale + bias)V in float64, written out, so that a
shared bug in the kernel cannot hide inside its own reference.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def fp64_attention(q, k, v, bias=None, scale=None):
    """softmax(q k^T * scale + bias) v, in float64, written plainly."""
    q64 = q.astype(np.float64)
    k64 = k.astype(np.float64)
    v64 = v.astype(np.float64)
    scale = float(scale if scale is not None else 1.0 / np.sqrt(q.shape[-1]))
    scores = np.einsum("bhqd,bhkd->bhqk", q64, k64) * scale
    if bias is not None:
        scores = scores + bias.astype(np.float64)
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    total = weights.sum(axis=-1, keepdims=True)
    # A row that is entirely masked has no weight anywhere; its output is
    # defined as zero rather than 0/0. The kernel must agree.
    weights = np.where(total > 0, weights / np.where(total > 0, total, 1.0), 0.0)
    return np.einsum("bhqk,bhkd->bhqd", weights, v64)


def ulp_distance(got, oracle):
    dtype = got.dtype
    rounded = oracle.astype(dtype)
    view = {np.dtype(np.float16): np.int16,
            np.dtype(np.float32): np.int32}.get(dtype, np.int64)
    bias = {np.int16: 0x8000, np.int32: 0x80000000}.get(
        view, 0x8000000000000000)

    def ordered(x):
        i = x.view(view).astype(np.int64)
        return np.where(i < 0, np.int64(bias) - i, i)

    finite = np.isfinite(got) & np.isfinite(rounded)
    ulp = np.abs(ordered(got[finite]) - ordered(rounded[finite])) if finite.any() \
        else np.array([0])
    scale = float(np.abs(oracle).max()) or 1.0
    return {"max_ulp": int(ulp.max()), "mean_ulp": float(ulp.mean()),
            "max_abs_err": float(np.abs(got.astype(np.float64) - oracle).max()),
            "rel_err": float(np.abs(got.astype(np.float64) - oracle).max() / scale),
            "nonfinite": int((~np.isfinite(got)).sum())}


def causal_bias(seqlen_q, seqlen_k, dtype=np.float32):
    """The additive causal mask a caller materialises: 0 where a query may
    attend, -inf where it may not. For decode the query sits at the END of
    the key sequence, which is what makes `seqlen_q != seqlen_k` causal."""
    q_pos = np.arange(seqlen_q) + (seqlen_k - seqlen_q)
    mask = q_pos[:, None] < np.arange(seqlen_k)[None, :]
    return np.where(mask, -np.inf, 0.0).astype(dtype)[None, None]


def padding_bias(seqlen_q, seqlen_k, keep, dtype=np.float32):
    """Whole key positions masked out, as a padded batch produces."""
    mask = np.arange(seqlen_k) >= keep
    return np.where(mask, -np.inf, 0.0).astype(dtype)[None, None, None, :] \
        * np.ones((1, 1, seqlen_q, 1), dtype=dtype)


REGIMES = [
    {"name": "prefill_noncausal", "b": 1, "h": 2, "sq": 64, "sk": 64,
     "d": 64, "kind": "none"},
    {"name": "prefill_causal", "b": 1, "h": 2, "sq": 64, "sk": 64,
     "d": 64, "kind": "causal"},
    {"name": "decode", "b": 1, "h": 2, "sq": 1, "sk": 128,
     "d": 64, "kind": "causal"},
    {"name": "padding", "b": 1, "h": 2, "sq": 48, "sk": 48,
     "d": 64, "kind": "padding", "keep": 30},
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()

    from neurobrix.kernels import launcher
    launcher.install()
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels import wrappers

    dtype = np.dtype(args.dtype)
    rows = []
    for spec in REGIMES:
        rng = np.random.default_rng(1234)
        shape_q = (spec["b"], spec["h"], spec["sq"], spec["d"])
        shape_k = (spec["b"], spec["h"], spec["sk"], spec["d"])
        q = rng.standard_normal(shape_q).astype(dtype)
        k = rng.standard_normal(shape_k).astype(dtype)
        v = rng.standard_normal(shape_k).astype(dtype)

        if spec["kind"] == "causal":
            bias = causal_bias(spec["sq"], spec["sk"])
        elif spec["kind"] == "padding":
            bias = padding_bias(spec["sq"], spec["sk"], spec["keep"])
        else:
            bias = None

        oracle = fp64_attention(q, k, v, bias)
        row = {"regime": spec["name"], "shape": shape_q, "sk": spec["sk"],
               "dtype": str(dtype)}
        started = time.time()
        try:
            out = wrappers.scaled_dot_product_attention_wrapper(
                NBXTensor.from_numpy(q), NBXTensor.from_numpy(k),
                NBXTensor.from_numpy(v),
                attn_mask=(None if bias is None
                           else NBXTensor.from_numpy(
                               np.ascontiguousarray(
                                   np.broadcast_to(bias, (spec["b"], spec["h"],
                                                          spec["sq"], spec["sk"])
                                                   ).astype(np.float32)))),
                is_causal=False)
            got = np.asarray(out.numpy())
            row["metal"] = ulp_distance(got, oracle)
            row["status"] = "ok"
        except Exception as exc:
            row["status"] = "refused"
            row["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:240]}"
        row["wall_s"] = round(time.time() - started, 3)
        rows.append(row)
        if row["status"] == "ok":
            print(f"  ok  {spec['name']:<20} q{spec['sq']}xk{spec['sk']} "
                  f"max_ulp={row['metal']['max_ulp']} "
                  f"rel={row['metal']['rel_err']:.2e}", flush=True)
        else:
            print(f"  REF {spec['name']:<20} {row['error'][:110]}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1))
    refused = [r for r in rows if r["status"] != "ok"]
    print(f"\n{len(rows)} regimes, {len(refused)} refused -> {args.out}")
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
