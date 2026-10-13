#!/usr/bin/env python
"""The Liger-Kernel fused kernels on this rack's cards, verified against ATen —
the portability datum the triton-ext thread asks for (the same Triton kernel,
verified on V100 here and on M4 Pro by the other machine, 2026-09-16).

Runs Liger's Triton kernels (RMSNorm, SwiGLU, GeGLU, RoPE) on the card given,
compares each with the ATen reference in fp32, and writes one JSON per run
carrying the environment (torch, triton, cuda, driver, card, python) beside
the numbers. torch is imported here on purpose: this is a diagnostic under
tools/, never under src/.

    python tools/liger_probe.py --out <campaign dir> [--card 0] [--rtol 1e-6]

The bound is RELATIVE to the tensor's magnitude, because an absolute one cannot
be met: one fp32 ulp at magnitude ten is about 1e-6, so a bound of 1e-7 there
asks two implementations to be bit-identical and calls every reordering a
failure. The first run of this probe (2026-09-16) returned FAIL on four of five
kernels that agree with ATen to two or three ulp. A verdict field is an
assertion like any other, and one that cannot be satisfied says nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time


def env_record(dev: int) -> dict:
    import torch, triton
    drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader", "-i", str(dev)],
                         capture_output=True, text=True).stdout.strip()
    return {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "triton": triton.__version__,
            "driver_and_card": drv, "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(dev))),
            "interpreter": sys.executable, "liger_kernel": __import__("liger_kernel").__version__ if hasattr(__import__("liger_kernel"), "__version__") else "?"}


def judge(max_abs_diff: float, magnitude: float, atol: float, rtol: float) -> str:
    """PASS when the difference fits the bound the hardware can actually meet.

    `|diff| <= atol + rtol * magnitude`, the standard mixed bound: rtol carries the
    reordering cost of a fused kernel against separate ops (1e-6 of the magnitude is
    about eight fp32 ulp), atol keeps a tensor whose magnitude is near zero from
    being judged against a bound near zero."""
    return "PASS" if max_abs_diff <= atol + rtol * magnitude else "FAIL"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--card", type=int, default=0)
    ap.add_argument("--rtol", type=float, default=1e-6,
                    help="accepted |diff| as a fraction of the tensor's magnitude (1e-6 is about 8 fp32 ulp there)")
    ap.add_argument("--atol", type=float, default=1e-6,
                    help="floor of the bound, for tensors whose magnitude is near zero")
    ap.add_argument("--tol", type=float, default=1e-7,
                    help="the other machine's ABSOLUTE bound, reported beside the verdict so the two files compare")
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--cols", type=int, default=4096)
    a = ap.parse_args()
    import torch
    torch.manual_seed(0)
    dev = torch.device(f"cuda:{a.card}")
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    from liger_kernel.ops.geglu import LigerGELUMulFunction
    from liger_kernel.ops.rope import LigerRopeFunction
    R, C = a.rows, a.cols
    results = []

    def cell(name, ref, got, shape):
        # Two numbers, and each says a different thing. `ulps_at_scale` is the
        # difference in ulps OF THE TENSOR'S MAGNITUDE — the reordering cost,
        # which is what a fused kernel against separate ops is being asked
        # about. `max_ulps_elementwise` divides by each element's own magnitude
        # and therefore explodes wherever the reference is near zero (RoPE
        # returned 15 410 on a 4.8e-7 difference), so it is recorded and never
        # judged on. The record also carries the other machine's ABSOLUTE
        # bound, met or not, so the two files can be compared as they are.
        diff = (ref.float() - got.float()).abs()
        d = diff.max().item()
        mag = ref.float().abs().max().item()
        eps = torch.finfo(torch.float32).eps
        ulp_at_scale = eps * mag
        elementwise = (diff / (eps * ref.float().abs().clamp(min=1e-30))).max().item()
        results.append({"kernel": name, "shape": list(shape), "dtype": "float32", "max_abs_diff": d,
                        "max_abs_ref": mag, "max_rel_diff": d / mag if mag else 0.0,
                        "ulps_at_scale": d / ulp_at_scale if ulp_at_scale else 0.0,
                        "max_ulps_elementwise": elementwise,
                        "verdict": judge(d, mag, a.atol, a.rtol),
                        "bound": f"|diff| <= {a.atol:.0e} + {a.rtol:.0e} * magnitude",
                        "meets_absolute_bound_of_the_other_machine": bool(d <= a.tol), "tol_abs": a.tol})

    x = torch.randn(R, C, device=dev, dtype=torch.float32)
    w = torch.randn(C, device=dev, dtype=torch.float32)
    eps = 1e-6
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
    got = LigerRMSNormFunction.apply(x, w, eps, 0.0, "llama", True)
    cell("rms_norm", ref, got, x.shape)
    a_ = torch.randn(R, C, device=dev); b_ = torch.randn(R, C, device=dev)
    cell("swiglu", torch.nn.functional.silu(a_) * b_, LigerSiLUMulFunction.apply(a_, b_), a_.shape)
    cell("geglu", torch.nn.functional.gelu(a_, approximate="tanh") * b_, LigerGELUMulFunction.apply(a_, b_), a_.shape)
    B, H, S, D = 2, 8, 128, 64
    q = torch.randn(B, H, S, D, device=dev); k = torch.randn(B, H, S, D, device=dev)
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2, device=dev).float() / D))
    t = torch.arange(S, device=dev).float()
    freqs = torch.outer(t, inv); emb = torch.cat([freqs, freqs], -1)
    cos, sin = emb.cos()[None], emb.sin()[None]
    def rot(x_):
        x1, x2 = x_[..., : D // 2], x_[..., D // 2:]
        return torch.cat([-x2, x1], -1)
    q_ref = q * cos[:, None] + rot(q) * sin[:, None]; k_ref = k * cos[:, None] + rot(k) * sin[:, None]
    q_got, k_got = LigerRopeFunction.apply(q, k, cos, sin)
    cell("rope_q", q_ref, q_got, q.shape); cell("rope_k", k_ref, k_got, k.shape)
    torch.cuda.synchronize(dev)
    rec = {"date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "env": env_record(a.card), "results": results,
           "verdict": "PASS" if all(r["verdict"] == "PASS" for r in results) else "FAIL"}
    os.makedirs(a.out, exist_ok=True)
    tag = f"{rec['env']['torch']}_triton{rec['env']['triton']}".replace("+", "_")
    path = os.path.join(a.out, f"liger_probe_v100_{tag}.json")
    json.dump(rec, open(path, "w"), indent=1)
    print(f"[liger probe] {rec['verdict']} on {rec['env']['driver_and_card']} — torch {rec['env']['torch']} triton {rec['env']['triton']}")
    for r in results:
        also = "" if r["meets_absolute_bound_of_the_other_machine"] else \
            f"; not under the other machine's absolute {r['tol_abs']:.0e}"
        print(f"   {r['kernel']:8s} {r['shape']} max|diff| {r['max_abs_diff']:.3e} (ref magnitude {r['max_abs_ref']:.2f}, "
              f"rel {r['max_rel_diff']:.2e}, {r['ulps_at_scale']:.1f} ulp of the magnitude) "
              f"{r['verdict']} — {r['bound']}{also}")
    print(f"   written: {path}")
    return 0 if rec["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
