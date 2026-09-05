#!/usr/bin/env python3
"""The kernel reference bank — the numerical gate of every port.

For every ATen op the dispatch layer implements with a house kernel, at the
engine's tiles (config/vendors/<vendor>/<arch>.yml block sizes) and their edges:
seeded inputs, the kernel's output on CUDA through the NBX wrapper, an fp64
oracle (the same ATen op in float64, torch — a diagnostic under tools/, off
the execution path), and the ULP distance of the output from the oracle
rounded to the output dtype. One `.npz` per kernel per shape under
`validation_outputs/kernel_reference_bank/<kernel>/<op>__<shape>.npz`
(inputs, output, oracle, ulp statistics, the launched kernels), an INDEX.md
with the coverage: which of the 280 kernels a run reached, which it did not.
Regenerable by this CLI; a port (Metal, ROCm) replays the inputs and compares.

    python tools/kernel_reference_bank.py generate [--out DIR] [--ops mm,softmax] [--only-missing]
    python tools/kernel_reference_bank.py index    [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
OUT_DEFAULT = REPO / "validation_outputs" / "kernel_reference_bank"
TRACE = "/tmp/nbx_bank_launch_trace.tsv"
os.environ["NBX_LAUNCH_TRACE"] = TRACE      # read once by the launcher at import: arm it first


# ---------------------------------------------------------------------------
# tiles: the engine's data
# ---------------------------------------------------------------------------

def tiles() -> dict:
    import yaml
    cfg = yaml.safe_load((REPO / "src/neurobrix/config/vendors/nvidia/volta.yml").read_text())
    bs = cfg.get("block_sizes") or {}
    return {"gemm": bs.get("gemm", {}), "bmm": bs.get("bmm", {}), "conv2d": bs.get("conv2d", {}),
            "default": int(bs.get("default", 1024)), "softmax_cap": int(bs.get("softmax_cap", 8192)),
            "layernorm_tile": int(bs.get("layernorm_tile", 1024)), "reduction": int(bs.get("reduction", 1024))}


def edges(t: int):
    """The tile, one past it, and a few tiles plus a remainder."""
    return sorted({t, t + 1, 3 * t + 5, max(1, t // 2 - 1)})


# ---------------------------------------------------------------------------
# generators: (name, shape tag, positional args, kwargs) — numpy inputs
# ---------------------------------------------------------------------------

def _n(rng, *shape, dtype=np.float16, scale=1.0):
    return (rng.standard_normal(shape) * scale).astype(dtype)


def gen_elementwise_unary(op, rng, T):
    for n in edges(T["default"]):
        x = _n(rng, n)
        if op in ("log", "sqrt", "rsqrt", "log1p"):
            x = np.abs(x) + np.float16(0.5)
        yield f"n{n}", (x,), {}


def gen_elementwise_binary(op, rng, T):
    for n in edges(T["default"]):
        a, b = _n(rng, n), _n(rng, n)
        if op == "div":
            b = np.where(np.abs(b) < 0.25, np.float16(0.5), b).astype(np.float16)
        if op == "pow":
            a = np.abs(a) + np.float16(0.5); b = np.full_like(a, 2.0)
        yield f"n{n}", (a, b), {}
    # broadcast over rows
    a = _n(rng, 7, T["default"] + 1); b = _n(rng, 1, T["default"] + 1)
    if op == "pow":
        a = np.abs(a) + np.float16(0.5); b = np.abs(b).astype(np.float16)   # a real result everywhere
    if op == "div":
        b = np.where(np.abs(b) < 0.25, np.float16(0.5), b).astype(np.float16)
    yield f"bcast7x{T['default'] + 1}", (a, b), {}


def gen_mm(op, rng, T):
    bm, bn, bk = int(T["gemm"].get("block_m", 64)), int(T["gemm"].get("block_n", 64)), int(T["gemm"].get("block_k", 32))
    for M, N, K in ((bm, bn, bk), (bm + 1, bn + 1, bk + 1), (3 * bm + 5, 2 * bn + 3, 4 * bk + 7), (1, bn, bk)):
        a, b = _n(rng, M, K, scale=0.5), _n(rng, K, N, scale=0.5)
        if op == "mm":
            yield f"M{M}N{N}K{K}", (a, b), {}
        elif op == "addmm":
            yield f"M{M}N{N}K{K}", (_n(rng, N), a, b), {}
        elif op == "bmm":
            yield f"B3M{M}N{N}K{K}", (_n(rng, 3, M, K, scale=0.5), _n(rng, 3, K, N, scale=0.5)), {}
        elif op == "baddbmm":
            yield f"B3M{M}N{N}K{K}", (_n(rng, 3, M, N), _n(rng, 3, M, K, scale=0.5), _n(rng, 3, K, N, scale=0.5)), {}
        elif op == "matmul":
            yield f"M{M}N{N}K{K}", (a, b), {}


def gen_softmax(op, rng, T):
    for n in (T["softmax_cap"] - 1, T["softmax_cap"], 257, 1024):
        if op == "_log_softmax":
            yield f"rows4x{n}", (_n(rng, 4, n, scale=3.0), -1), {}
        else:
            yield f"rows4x{n}", (_n(rng, 4, n, scale=3.0), -1, False), {}


def gen_layer_norm(op, rng, T):
    for h in edges(T["layernorm_tile"]):
        x = _n(rng, 5, h)
        yield f"rows5x{h}", (x, [h], _n(rng, h, scale=0.3) + np.float16(1.0), _n(rng, h, scale=0.1), 1e-5), {}


def gen_rms_norm(op, rng, T):
    for h in edges(T["layernorm_tile"]):
        yield f"rows5x{h}", (_n(rng, 5, h), _n(rng, h, scale=0.3) + np.float16(1.0), 1e-6), {}


def gen_group_norm(op, rng, T):
    for C, HW in ((32, 33), (64, 257)):
        x = _n(rng, 2, C, HW, 1)
        yield f"C{C}HW{HW}G8", (x, _n(rng, C, scale=0.3) + np.float16(1.0), _n(rng, C, scale=0.1), 2, C, HW, 8, 1e-5), {}


def gen_reduction(op, rng, T):
    for n in edges(T["reduction"]):
        x = _n(rng, 3, n)
        if op in ("sum", "mean", "amax", "amin"):
            yield f"rows3x{n}_dim1", (x, [1], False), {}
        elif op in ("max", "min", "argmax", "argmin"):
            yield f"rows3x{n}_dim1", (x, 1, False), {}
        elif op == "cumsum":
            yield f"rows3x{n}_dim1", (x, 1), {}


def gen_conv2d(op, rng, T):
    bh, bw = int(T["conv2d"].get("block_h", 16)), int(T["conv2d"].get("block_w", 16))
    for (N, C, H, W, O, k) in ((1, 4, bh, bw, 8, 3), (2, 3, bh + 1, bw + 5, 5, 3), (1, 8, 2 * bh + 3, 2 * bw + 3, 8, 1)):
        yield f"N{N}C{C}H{H}W{W}O{O}k{k}", (_n(rng, N, C, H, W), _n(rng, O, C, k, k, scale=0.3), _n(rng, O, scale=0.1),
                                            [1, 1], [k // 2, k // 2], [1, 1], False, [0, 0], 1), {}


def gen_sdpa(op, rng, T):
    for (B, H, S, D) in ((1, 2, 64, 64), (1, 2, 65, 64), (2, 4, 257, 64)):
        q, k, v = _n(rng, B, H, S, D, scale=0.3), _n(rng, B, H, S, D, scale=0.3), _n(rng, B, H, S, D, scale=0.3)
        if op == "_scaled_dot_product_efficient_attention":
            yield f"B{B}H{H}S{S}D{D}", (q, k, v, None, False, 0.0, False), {}
        else:
            yield f"B{B}H{H}S{S}D{D}", (q, k, v), {}


def gen_embedding(op, rng, T):
    w = _n(rng, 100, T["default"] + 1)
    idx = rng.integers(0, 100, size=(2, 17)).astype(np.int64)
    yield f"V100D{T['default'] + 1}", (w, idx), {}


def gen_index_select(op, rng, T):
    x = _n(rng, 33, T["default"] + 1)
    idx = rng.integers(0, 33, size=(19,)).astype(np.int64)
    yield f"rows33_sel19", (x, 0, idx), {}


def gen_cat(op, rng, T):
    yield f"cat3", ([_n(rng, 3, 5), _n(rng, 2, 5), _n(rng, 4, 5)], 0), {}


def gen_pad(op, rng, T):
    yield f"pad2d", (_n(rng, 2, 3, 9, 11), [1, 2, 3, 4], 0.0), {}


def gen_upsample(op, rng, T):
    yield f"x2", (_n(rng, 1, 3, 9, 11), [18, 22]), {}


def gen_where(op, rng, T):
    a, b = _n(rng, 1025), _n(rng, 1025)
    yield f"n1025", ((a > 0), a, b), {}


def gen_clamp(op, rng, T):
    yield f"n1025", (_n(rng, 1025, scale=2.0), -1.0, 1.0), {}


GENERATORS = {
    **{o: gen_elementwise_unary for o in ("exp", "log", "sin", "cos", "tanh", "sigmoid", "gelu", "silu", "relu",
                                          "sqrt", "rsqrt", "neg", "abs", "erf", "log1p",
                                          "floor", "ceil", "trunc")},
    **{o: gen_elementwise_binary for o in ("add", "sub", "mul", "div", "pow", "maximum", "minimum", "remainder")},
    **{o: gen_mm for o in ("mm", "addmm", "bmm", "baddbmm", "matmul")},
    "_softmax": gen_softmax, "_log_softmax": gen_softmax,
    "native_layer_norm": gen_layer_norm, "rms_norm": gen_rms_norm, "native_group_norm": gen_group_norm,
    **{o: gen_reduction for o in ("sum", "mean", "amax", "amin", "max", "min", "argmax", "argmin", "cumsum")},
    "convolution": gen_conv2d,
    "_scaled_dot_product_efficient_attention": gen_sdpa, "_scaled_dot_product_flash_attention": gen_sdpa,
    "scaled_dot_product_attention": gen_sdpa,
    "embedding": gen_embedding, "index_select": gen_index_select, "cat": gen_cat,
    "constant_pad_nd": gen_pad, "upsample_nearest2d": gen_upsample, "where": gen_where, "clamp": gen_clamp,
}


# ---------------------------------------------------------------------------
# the two sides
# ---------------------------------------------------------------------------

def to_nbx(x):
    from neurobrix.kernels.nbx_tensor import NBXTensor
    if isinstance(x, np.ndarray):
        return NBXTensor.from_numpy(np.ascontiguousarray(x))
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], np.ndarray):
        return [to_nbx(e) for e in x]
    return x


def to_torch64(x, device):
    import torch
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(x)).to(device)
        return t.double() if t.is_floating_point() else t
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], np.ndarray):
        return [to_torch64(e, device) for e in x]
    return x


def from_nbx(t):
    import numpy as _np
    if hasattr(t, "numpy"):
        return t.numpy()
    if isinstance(t, (tuple, list)):
        return [from_nbx(e) for e in t]
    return t


def ulp_distance(out: np.ndarray, oracle: np.ndarray) -> np.ndarray:
    """|ULPs| between `out` and the oracle rounded to out's dtype (integer
    view of the float bits, sign-folded to a monotone ordinal)."""
    dt = out.dtype
    ref = oracle.astype(dt)
    ints = {np.dtype("float16"): np.int16, np.dtype("float32"): np.int32, np.dtype("float64"): np.int64}[dt]
    a = out.view(ints).astype(np.int64); b = ref.view(ints).astype(np.int64)
    bias = {np.int16: 1 << 15, np.int32: 1 << 31, np.int64: 1 << 63}[ints]
    a = np.where(a < 0, bias - a, a); b = np.where(b < 0, bias - b, b)   # sign-magnitude → ordinal
    return np.abs(a - b)


# The oracle op when torch's own name differs (the fused attention variants take
# extra arguments the composite op does not).
ORACLE = {"_scaled_dot_product_efficient_attention": "scaled_dot_product_attention",
          "_scaled_dot_product_flash_attention": "scaled_dot_product_attention"}


def _oracle_rms_norm(x, weight, eps=1e-6, epsilon=None):
    import torch
    e = eps if epsilon is None else epsilon
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + e) * weight


ORACLE_FN = {"rms_norm": _oracle_rms_norm}
# Extra oracle arguments torch's op takes that the wrapper does not.
ORACLE_EXTRA = {"_log_softmax": (False,)}
# Kernels this card cannot run (the wrapper refuses loudly): recorded, not counted as errors.
SKIP_ON_SM70 = {"_scaled_dot_product_flash_attention"}


def one_case(op: str, tag: str, args, kwargs, device: str, out_dir: Path):
    import torch
    wrapper = __import__("neurobrix.kernels.dispatch", fromlist=["dispatch"]).dispatch(f"aten::{op}")
    if wrapper is None:
        return {"op": op, "tag": tag, "status": "no wrapper"}
    Path(TRACE).unlink(missing_ok=True)
    nbx_args = [to_nbx(a) for a in args]
    t0 = time.time()
    try:
        out = wrapper(*nbx_args, **kwargs)
    except Exception as e:
        return {"op": op, "tag": tag, "status": "wrapper error", "error": f"{type(e).__name__}: {str(e)[:200]}"}
    torch.cuda.synchronize()
    wall = time.time() - t0
    launched = sorted({l.split("\t")[0] for l in open(TRACE)} if os.path.exists(TRACE) else set())
    outs = from_nbx(out)
    outs = outs if isinstance(outs, list) else [outs]
    outs = [o for o in outs if isinstance(o, np.ndarray)]
    aten = ORACLE_FN.get(op) or getattr(torch.ops.aten, ORACLE.get(op, op))
    oracle_args = (args[:3] if op in ORACLE else args) + ORACLE_EXTRA.get(op, ())
    try:
        ref = aten(*[to_torch64(a, device) for a in oracle_args], **kwargs)
    except Exception as e:
        return {"op": op, "tag": tag, "status": "oracle error", "launched": launched, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    refs = list(ref) if isinstance(ref, (tuple, list)) else [ref]
    refs = [r.detach().cpu().numpy() for r in refs if torch.is_tensor(r)]
    stats = []
    arrays = {}
    for i, (o, r) in enumerate(zip(outs, refs)):
        if o.shape != r.shape:
            stats.append({"out": i, "status": f"shape {o.shape} vs oracle {r.shape}"}); continue
        arrays[f"out{i}"] = o; arrays[f"oracle{i}"] = r
        if o.dtype.kind == "f":
            u = ulp_distance(o, r)
            finite = np.isfinite(o) & np.isfinite(r)
            err = np.abs(o.astype(np.float64) - r)
            scale = float(np.abs(r[finite]).max()) if finite.any() else 0.0
            max_err = float(err[finite].max()) if finite.any() else None
            stats.append({"out": i, "dtype": str(o.dtype), "max_ulp": int(u[finite].max()) if finite.any() else None,
                          "mean_ulp": float(u[finite].mean()) if finite.any() else None,
                          "max_abs_err": max_err, "oracle_scale": scale,
                          # the headline: the largest error relative to the output's scale — an
                          # ULP count explodes across zero after a cancellation and says nothing there
                          "rel_err": (max_err / scale) if (max_err is not None and scale > 0) else None,
                          "nonfinite": int((~finite).sum()), "identical": bool(np.array_equal(o, r.astype(o.dtype)))})
        else:
            stats.append({"out": i, "dtype": str(o.dtype), "identical": bool(np.array_equal(o, r))})
    for i, a in enumerate(args):
        if isinstance(a, np.ndarray):
            arrays[f"in{i}"] = a
    kernel = launched[0] if launched else "_unlaunched"
    d = out_dir / kernel
    d.mkdir(parents=True, exist_ok=True)
    meta = {"op": op, "tag": tag, "kwargs": {k: str(v) for k, v in kwargs.items()}, "launched": launched,
            "stats": stats, "wall_s": wall, "status": "ok"}
    np.savez(d / f"{op}__{tag}.npz", meta=json.dumps(meta), **arrays)
    return meta


def generate(out: Path, ops, only_missing: bool):
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    from neurobrix.kernels.dispatch import list_kernels
    DeviceAllocator.set_device(0)
    device = "cuda:0"
    T = tiles()
    available = set(list_kernels())
    todo = ops or sorted(k for k in GENERATORS if k in available)
    absent = sorted(k for k in GENERATORS if k not in available)
    results = []
    for op in todo:
        gen = GENERATORS.get(op)
        if gen is None:
            print(f"[bank] {op}: no generator"); continue
        if op in SKIP_ON_SM70 and T.get("arch", 70) < 80:
            results.append({"op": op, "tag": "*", "status": "not on this card (sm_70)"}); print(f"[bank] {op}: not on this card (sm_70)"); continue
        rng = np.random.default_rng(abs(hash(op)) % (2 ** 32))
        for tag, args, kwargs in gen(op, rng, T):
            if only_missing and list(out.glob(f"*/{op}__{tag}.npz")):
                continue
            try:
                r = one_case(op, tag, args, kwargs, device, out)
            except Exception as e:
                r = {"op": op, "tag": tag, "status": "crash", "error": f"{type(e).__name__}: {str(e)[:200]}\n{traceback.format_exc()[-400:]}"}
            results.append(r)
            s = r.get("stats") or [r.get("status")]
            print(f"[bank] {op:<40} {tag:<24} {r.get('status')} {r.get('launched', '')} {s[0] if s else ''}"[:220], flush=True)
    (out / "generate_log.json").write_text(json.dumps({"results": results, "generators_without_wrapper": absent}, indent=1))
    return results


def index(out: Path) -> str:
    import ast
    kernels = {}
    for p in sorted((REPO / "src/neurobrix").rglob("*.py")):
        if "triton_kernels_ref" in str(p):
            continue
        try:
            tree = ast.parse(p.read_text())
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and any(ast.unparse(d.func if isinstance(d, ast.Call) else d).endswith("jit") for d in n.decorator_list):
                kernels[n.name] = str(p.relative_to(REPO / "src/neurobrix"))
    reached = {}
    rows = []
    for f in sorted(out.glob("*/*.npz")):
        meta = json.loads(str(np.load(f, allow_pickle=False)["meta"]))
        for k in meta.get("launched", []):
            reached.setdefault(k, []).append(meta["op"])
        st = meta.get("stats") or []
        s0 = st[0] if st else {}
        rel = s0.get('rel_err')
        rows.append(f"| {meta['op']} | {meta['tag']} | {', '.join(meta.get('launched', [])) or '—'} | "
                    f"{s0.get('dtype', '')} | {'' if rel is None else f'{rel:.2e}'} | {s0.get('max_abs_err', '')} | {s0.get('max_ulp', '')} | {s0.get('identical', '')} | {f.relative_to(out)} |")
    unreached = sorted(k for k in kernels if k not in reached)
    text = ["# Kernel reference bank — INDEX", "",
            f"Kernels in the library: {len(kernels)}. Reached by the bank: {len(reached)}. Not reached: {len(unreached)}.",
            "Every `.npz` holds the seeded inputs, the kernel's output, the fp64 oracle, and its `meta` (ULP statistics, launched kernels).", "",
            "| op | shape | kernel(s) launched | out dtype | max rel err (vs output scale) | max abs err | max ULP (explodes across zero) | bit-identical to rounded oracle | file |", "|---|---|---|---|---|---|---|---|---|"] + rows
    text += ["", "## Kernels not reached by any reference yet (a generator to add, or an unlaunched kernel)", ""]
    text += [f"- `{k}` ({kernels[k]})" for k in unreached]
    (out / "INDEX.md").write_text("\n".join(text) + "\n")
    return f"reached {len(reached)}/{len(kernels)} kernels, {len(rows)} references"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate"); g.add_argument("--out", default=str(OUT_DEFAULT)); g.add_argument("--ops", default=""); g.add_argument("--only-missing", action="store_true")
    i = sub.add_parser("index"); i.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "generate":
        generate(out, [o for o in args.ops.split(",") if o], args.only_missing)
    print(index(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
