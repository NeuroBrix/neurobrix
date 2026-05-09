"""Phase 3a-bis — wrapper-level microtests with REAL captured runtime inputs.

Phase 3a tested the 9 VAE TOP-divergent ops with random fp16 inputs at
runtime shapes — all bit-exact. Phase 3a-bis replays the SAME wrappers
on REAL captured runtime tensors (saved by vae_isolation_probe.py with
NBX_CAPTURE_VAE_OPS=1 during the v5 garbage-producing decode).

If a wrapper is bit-exact on random but DIVERGENT on the real captured
input, that's the bug source — input-dependent: NaN/Inf/edge-case
patterns in the real activation distribution that random fp16 doesn't
reproduce.

If all wrappers are bit-exact on real inputs too, the bug is elsewhere
(stateful runtime side-effect, or some integration issue between the
op-uid_interceptors and the kernel that doesn't fire in standalone
wrapper invocation).
"""
import argparse
from pathlib import Path
import sys

ROOT = Path("/home/mlops/NeuroBrix_System")
DUMP_DIR = ROOT / "validation_outputs/p_sana_4kpx_runtime/audit_spaghetti_2026_05_05/vae_op_input_dumps"
OUT_MD = ROOT / "validation_outputs/p_sana_4kpx_runtime/audit_spaghetti_2026_05_05/microtest_vae_real_inputs_report.md"

sys.path.insert(0, str(ROOT / "src"))


def replay_op(uid: str, payload: dict, device: str = "cuda:2"):
    """Replay one op with captured inputs vs torch reference."""
    import torch
    import torch.nn.functional as F
    from neurobrix.kernels.nbx_tensor import NBXTensor, nbx_to_torch

    base = uid.split("::")[0].replace("aten.", "")

    # Reconstruct args. Each arg in payload["args"] is either a torch
    # tensor dict or a scalar.
    args_torch = []
    for a in payload.get("args", []):
        if a.get("kind") == "torch":
            t = a["tensor"]
            if not isinstance(t, torch.Tensor):
                continue
            t = t.to(device).contiguous()
            args_torch.append(t)
        elif a.get("kind") == "scalar":
            args_torch.append(a.get("value"))

    if not args_torch:
        return {"uid": uid, "error": "no torch args reconstructed"}

    # Build NBX inputs
    args_nbx = []
    for t in args_torch:
        if isinstance(t, torch.Tensor):
            arr = t.detach().cpu().numpy()
            args_nbx.append(NBXTensor.from_numpy(arr))
        else:
            args_nbx.append(t)

    # Dispatch by base name
    try:
        if base == "relu":
            from neurobrix.kernels.wrappers import relu as nbx_op
            nbx_out = nbx_op(args_nbx[0])
            ref_out = F.relu(args_torch[0])
        elif base == "silu":
            from neurobrix.kernels.wrappers import silu as nbx_op
            nbx_out = nbx_op(args_nbx[0])
            ref_out = F.silu(args_torch[0])
        elif base == "pixel_shuffle":
            from neurobrix.kernels.wrappers import pixel_shuffle_wrapper
            # pixel_shuffle takes (input, upscale_factor)
            upscale = 2
            if len(args_torch) > 1 and isinstance(args_torch[1], int):
                upscale = args_torch[1]
            nbx_out = pixel_shuffle_wrapper(args_nbx[0], upscale)
            ref_out = F.pixel_shuffle(args_torch[0], upscale)
        elif base == "convolution":
            from neurobrix.kernels.wrappers import conv2d_wrapper
            # Convolution args: (input, weight, bias, stride, padding, ...)
            x_t = args_torch[0]
            w_t = args_torch[1] if len(args_torch) > 1 else None
            x_nbx = args_nbx[0]
            w_nbx = args_nbx[1] if len(args_nbx) > 1 else None
            # Default stride/padding for typical 3x3 VAE conv
            nbx_out = conv2d_wrapper(x_nbx, w_nbx, stride=1, padding=1)
            ref_out = F.conv2d(x_t, w_t, stride=1, padding=1)
        else:
            return {"uid": uid, "error": f"unknown base op: {base}"}
    except Exception as e:
        import traceback
        return {"uid": uid, "error": f"{type(e).__name__}: {e}",
                "tb": traceback.format_exc()[-500:]}

    # Compare
    out_t = nbx_to_torch(nbx_out)
    ref_t = ref_out.detach().to(out_t.device, dtype=out_t.dtype)
    diff = (out_t.float() - ref_t.float()).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    ref_max = float(ref_t.abs().float().max())
    rel_max = max_abs / max(ref_max, 1e-9)

    return {
        "uid": uid, "base": base,
        "input_shape": tuple(args_torch[0].shape) if isinstance(args_torch[0], torch.Tensor) else None,
        "input_dtype": str(args_torch[0].dtype) if isinstance(args_torch[0], torch.Tensor) else None,
        "max_abs": max_abs, "mean_abs": mean_abs,
        "rel_max": rel_max, "ref_max": ref_max,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    import torch

    files = sorted(DUMP_DIR.glob("*.pt"))
    print(f"Found {len(files)} captured op dumps in {DUMP_DIR}")

    results = []
    for f in files:
        uid = f.stem.replace("_", ".").replace("..", "::").replace("..", "::")
        # filename pattern: aten_relu__15.pt -> aten_relu__15 -> aten.relu::15
        # double-underscore separates op name from occurrence
        parts = f.stem.split("__")
        if len(parts) == 2:
            uid = parts[0].replace("_", ".") + "::" + parts[1]
        if args.only and uid != args.only:
            continue
        print(f"\n[real-input-microtest] {uid} ({f.name}, {f.stat().st_size} bytes)")
        payload = torch.load(f, weights_only=False)
        actual_uid = payload.get("uid", uid)
        r = replay_op(actual_uid, payload, args.device)
        results.append(r)
        if "error" in r:
            print(f"  ERROR: {r['error']}")
        else:
            verdict = "BIT-EXACT" if r["max_abs"] == 0 else (
                "fp16 ULP" if r["max_abs"] < 1e-2 and r["rel_max"] < 0.01
                else "DIVERGENT"
            )
            print(f"  shape={r['input_shape']} max_abs={r['max_abs']:.6g} "
                  f"rel_max={r['rel_max']:.6g} verdict={verdict}")

    out_lines = []
    out_lines.append("# VAE TOP-divergent ops microtest — REAL runtime inputs (Phase 3a-bis)")
    out_lines.append("")
    out_lines.append("Replay of NBX wrappers on tensors captured at the corresponding op_uid")
    out_lines.append("during the v5 triton VAE garbage decode (`NBX_CAPTURE_VAE_OPS=1`)")
    out_lines.append("vs `torch.nn.functional` reference applied to the SAME captured input.")
    out_lines.append("")
    out_lines.append("| op_uid | base | input shape | input dtype | max_abs | rel_max | verdict |")
    out_lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        if "error" in r:
            out_lines.append(f"| `{r['uid']}` | — | — | — | — | — | ERROR: {r['error']} |")
            continue
        verdict = "BIT-EXACT" if r["max_abs"] == 0 else (
            "fp16 ULP" if r["max_abs"] < 1e-2 and r["rel_max"] < 0.01
            else "DIVERGENT"
        )
        out_lines.append(
            f"| `{r['uid']}` | {r['base']} | "
            f"{r.get('input_shape')} | {r.get('input_dtype')} | "
            f"{r['max_abs']:.4g} | {r['rel_max']:.4g} | {verdict} |"
        )
    out_lines.append("")

    divergent = [r for r in results
                 if "error" not in r and r["max_abs"] >= 1e-2]
    if divergent:
        out_lines.append("## **DIVERGENT ops (bug source on real inputs)**")
        out_lines.append("")
        for r in divergent:
            out_lines.append(f"- `{r['uid']}`: max_abs={r['max_abs']:.4g}, rel_max={r['rel_max']:.4g}")
        out_lines.append("")
        out_lines.append("These wrappers compute correctly on random fp16 (Phase 3a) but DIVERGENT on real captured runtime input. The bug is input-dependent — likely NaN/Inf/edge-case patterns in the actual activation distribution.")
    else:
        out_lines.append("## All ops bit-exact on real captured inputs too")
        out_lines.append("")
        out_lines.append("If random fp16 (Phase 3a) AND real runtime captures both produce bit-exact results from these wrappers, the bug is NOT in any of these 9 op kernels in any standalone invocation. The bug must be in the runtime pipeline integration: a side-effect of the dispatcher, the op_uid_interceptors registration timing, the kill_slots / deferred-free pool state, or something else that only fires when invoked from inside `vae_exec.run()` orchestration.")

    OUT_MD.write_text("\n".join(out_lines))
    print(f"\nreport: {OUT_MD}")


if __name__ == "__main__":
    main()
