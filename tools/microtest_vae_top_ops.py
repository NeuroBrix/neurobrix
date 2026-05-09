"""Phase 3a — wrapper-level microtests of VAE TOP-divergent ops at exact 4Kpx shapes.

Generates random fp16 inputs at the EXACT shapes that the VAE op
encounters during runtime (read from VAE graph.json), invokes the
NeuroBrix wrapper, compares to torch.nn.functional reference. If
any op diverges from torch reference at the bit level (or beyond
fp16 ULP for matmul-class), that op is the localized bug source.

Cas A1 (all bit-exact): bug is input-dependent; pivot to Phase 3a-bis
   with real runtime-captured intermediate inputs.
Cas A2 (>=1 divergent): kernel itself is buggy at this shape; audit
   that wrapper / kernel.

Targets from cross-variant analysis (Sana 4Kpx vs Sana 1024 in
diff_dag_cross_variant_report.md, rel_ratio extreme):
- aten.relu::15        (rel_ratio 2.88M× — the most extreme)
- aten.silu::18..23    (rel_ratio 165× to 23k×)
- aten.pixel_shuffle::3 (rel_ratio 504×)
- aten.convolution::61 (rel_ratio 19×)

No NeuroBrix code modification. Pure consumer of public wrappers.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/home/mlops/NeuroBrix_System")
VAE_GRAPH = (ROOT / ".." / ".." / "home/mlops/.neurobrix/cache/"
             "Sana_1600M_4Kpx_BF16/components/vae/graph.json").resolve()
# Direct path
VAE_GRAPH = Path("/home/mlops/.neurobrix/cache/Sana_1600M_4Kpx_BF16"
                 "/components/vae/graph.json")
OUT_DIR = ROOT / "validation_outputs/p_sana_4kpx_runtime/audit_spaghetti_2026_05_05"
OUT_MD = OUT_DIR / "microtest_vae_top_ops_report.md"

sys.path.insert(0, str(ROOT / "src"))

# Targets — verified from VAE graph.json:
# (op_uid, kernel_kind, [extra_kwargs])
# kernel_kind defines which wrapper to invoke.
TARGETS = [
    ("aten.relu::15",         "relu",          {}),
    ("aten.silu::18",         "silu",          {}),
    ("aten.silu::19",         "silu",          {}),
    ("aten.silu::20",         "silu",          {}),
    ("aten.silu::21",         "silu",          {}),
    ("aten.silu::22",         "silu",          {}),
    ("aten.silu::23",         "silu",          {}),
    ("aten.pixel_shuffle::3", "pixel_shuffle", {"upscale_factor": 2}),
    ("aten.convolution::61",  "conv2d",        {}),
]


def load_op_by_uid(uid: str):
    g = json.load(VAE_GRAPH.open())
    ops = g.get("ops") or g.get("operations")
    if isinstance(ops, dict):
        ops = list(ops.values())
    for o in ops:
        if o.get("op_uid") == uid:
            return o
    raise KeyError(uid)


def gen_input(shape, dtype, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal(tuple(shape)).astype("float32")
    if dtype == "float16":
        return arr.astype("float16")
    if dtype == "bfloat16":
        # Need to use torch for bf16 (numpy has no native bf16)
        import torch
        return torch.from_numpy(arr).to(torch.bfloat16)
    return arr


def to_torch_cuda(x, target_device="cuda:0"):
    import numpy as np
    import torch
    if isinstance(x, torch.Tensor):
        return x.to(target_device).contiguous()
    return torch.from_numpy(np.ascontiguousarray(x)).to(target_device)


def to_nbx(x):
    """Move torch.Tensor to NBXTensor via numpy."""
    import torch
    from neurobrix.kernels.nbx_tensor import NBXTensor
    arr = x.detach().cpu().numpy()
    return NBXTensor.from_numpy(arr)


def from_nbx(x):
    from neurobrix.kernels.nbx_tensor import nbx_to_torch
    return nbx_to_torch(x)


def run_microtest(uid: str, kind: str, extra_kw: dict, target_dev: str):
    import torch
    import torch.nn.functional as F

    op = load_op_by_uid(uid)
    in_shapes = op.get("input_shapes")
    in_dtypes = op.get("input_dtypes") or ["float32"]
    out_shapes = op.get("output_shapes")

    # Match graph dtype if it's float-something (fp32 in graph means
    # graph DAG records the trace-time precision; on Volta runtime
    # these are downcast to fp16 by Prism for non-matmul ops). We'll
    # test in fp16 to mirror the actual triton path on Volta.
    test_dtype = "float16"

    # Generate first input tensor (the activation feeding the op).
    seed = hash(uid) & 0xFFFF
    x_torch = to_torch_cuda(
        gen_input(in_shapes[0], test_dtype, seed), target_dev
    )
    x_nbx = to_nbx(x_torch)

    # Invoke NBX wrapper + torch reference.
    if kind == "relu":
        from neurobrix.kernels.wrappers import relu as nbx_relu
        nbx_out = nbx_relu(x_nbx)
        ref_out = F.relu(x_torch)
    elif kind == "silu":
        from neurobrix.kernels.wrappers import silu as nbx_silu
        nbx_out = nbx_silu(x_nbx)
        ref_out = F.silu(x_torch)
    elif kind == "pixel_shuffle":
        from neurobrix.kernels.wrappers import pixel_shuffle_wrapper
        upscale = extra_kw.get("upscale_factor", 2)
        nbx_out = pixel_shuffle_wrapper(x_nbx, upscale)
        # Note: graph records NHWC-style (1, 1024, 1024, 1024). Torch's
        # pixel_shuffle expects NCHW. The op_uid records the user-facing
        # ATen op. To match torch's NCHW convention, we permute the
        # NHWC layout: input shape (1, 1024, 1024, 1024) treated as
        # NCHW means C=1024 H=1024 W=1024 — which is what we have.
        ref_out = F.pixel_shuffle(x_torch, upscale)
    elif kind == "conv2d":
        from neurobrix.kernels.wrappers import conv2d_wrapper
        # Need weight tensor too (shape from graph input_shapes[1]).
        w_shape = in_shapes[1]
        w_torch = to_torch_cuda(
            gen_input(w_shape, test_dtype, seed + 1), target_dev
        )
        w_nbx = to_nbx(w_torch)
        # Default conv params from graph attributes
        attrs = op.get("attributes", {}) or {}
        kwargs = attrs.get("kwargs", {}) or {}
        padding = kwargs.get("padding", [1, 1])
        if isinstance(padding, dict):
            padding = padding.get("value", [1, 1])
        stride = kwargs.get("stride", [1, 1])
        if isinstance(stride, dict):
            stride = stride.get("value", [1, 1])
        nbx_out = conv2d_wrapper(x_nbx, w_nbx, stride=stride, padding=padding)
        ref_out = F.conv2d(x_torch, w_torch, stride=stride, padding=padding)
    else:
        raise NotImplementedError(kind)

    # Compare.
    out_t = from_nbx(nbx_out)
    ref_t = ref_out.detach().to(out_t.device, dtype=out_t.dtype)
    diff = (out_t.float() - ref_t.float()).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    ref_max = float(ref_t.abs().float().max())
    rel_max = max_abs / max(ref_max, 1e-9)

    # Sample 5 positions for reporting.
    n = ref_t.numel()
    idxs = (0, n // 8, n // 4, n // 2, 3 * n // 4)
    ref_samples = [float(ref_t.flatten()[i].cpu()) for i in idxs]
    nbx_samples = [float(out_t.flatten()[i].cpu()) for i in idxs]

    return {
        "uid": uid, "kind": kind,
        "in_shapes": in_shapes, "out_shape": tuple(out_t.shape),
        "max_abs": max_abs, "mean_abs": mean_abs,
        "rel_max": rel_max, "ref_max": ref_max,
        "ref_samples": ref_samples, "nbx_samples": nbx_samples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only", default=None,
                    help="run only this op_uid (e.g. aten.silu::18)")
    args = ap.parse_args()

    results = []
    for uid, kind, extra_kw in TARGETS:
        if args.only and uid != args.only:
            continue
        print(f"\n[microtest] {uid} ({kind})", flush=True)
        try:
            r = run_microtest(uid, kind, extra_kw, args.device)
            results.append(r)
            verdict = "BIT-EXACT" if r["max_abs"] == 0 else (
                "fp16 ULP" if r["max_abs"] < 1e-2 and r["rel_max"] < 0.01
                else "DIVERGENT"
            )
            print(f"  shape={r['in_shapes'][0]} max_abs={r['max_abs']:.6g} "
                  f"rel_max={r['rel_max']:.6g} verdict={verdict}", flush=True)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            results.append({"uid": uid, "kind": kind, "error": str(e)})

    # Markdown report.
    out_lines = []
    out_lines.append("# VAE TOP-divergent ops microtest (Phase 3a, wrapper-level, fp16 random)")
    out_lines.append("")
    out_lines.append("Bit-exact check of NBX wrappers vs torch.nn.functional at exact shapes from VAE graph.json.")
    out_lines.append("")
    out_lines.append("| op_uid | kind | input shape | max_abs | rel_max | verdict |")
    out_lines.append("|---|---|---|---|---|---|")
    for r in results:
        if "error" in r:
            out_lines.append(f"| `{r['uid']}` | {r['kind']} | — | — | — | ERROR: {r['error']} |")
            continue
        verdict = "BIT-EXACT" if r["max_abs"] == 0 else (
            "fp16 ULP" if r["max_abs"] < 1e-2 and r["rel_max"] < 0.01
            else "DIVERGENT"
        )
        out_lines.append(
            f"| `{r['uid']}` | {r['kind']} | "
            f"{r['in_shapes'][0] if r.get('in_shapes') else '?'} | "
            f"{r['max_abs']:.4g} | {r['rel_max']:.4g} | {verdict} |"
        )
    out_lines.append("")

    out_lines.append("## Verdict synthesis")
    out_lines.append("")
    divergent = [r for r in results
                 if "error" not in r and r["max_abs"] >= 1e-2]
    bit_exact = [r for r in results
                 if "error" not in r and r["max_abs"] == 0]
    fp16_noise = [r for r in results
                  if "error" not in r and 0 < r["max_abs"] < 1e-2
                  and r["rel_max"] < 0.01]
    errors = [r for r in results if "error" in r]
    out_lines.append(f"- bit-exact (max_abs == 0): {len(bit_exact)}")
    out_lines.append(f"- fp16 ULP (max_abs < 1e-2 AND rel_max < 1%): {len(fp16_noise)}")
    out_lines.append(f"- DIVERGENT (max_abs >= 1e-2): {len(divergent)}")
    out_lines.append(f"- errors: {len(errors)}")
    out_lines.append("")
    if divergent:
        out_lines.append("### DIVERGENT ops (Cas A2 — kernel-level bug localized)")
        out_lines.append("")
        for r in divergent:
            out_lines.append(f"- `{r['uid']}` ({r['kind']}): max_abs={r['max_abs']:.4g}, rel_max={r['rel_max']:.4g}")
            out_lines.append(f"  - ref_samples = {r['ref_samples']}")
            out_lines.append(f"  - nbx_samples = {r['nbx_samples']}")
        out_lines.append("")
    elif not errors:
        out_lines.append("### Cas A1 — all ops bit-exact in isolation")
        out_lines.append("")
        out_lines.append("Random fp16 inputs do not trigger the bug. Pivot to Phase 3a-bis: "
                         "instrument vae_isolation_probe.py to dump the actual runtime "
                         "intermediate inputs at these op_idx during triton VAE garbage decode, "
                         "then re-run microtests with those captured tensors. The bug may be "
                         "input-dependent (NaN/Inf/edge cases not triggered by random).")
    out_lines.append("")

    OUT_MD.write_text("\n".join(out_lines))
    print(f"\nreport: {OUT_MD}")


if __name__ == "__main__":
    main()
