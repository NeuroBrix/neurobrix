#!/usr/bin/env python3
"""Our decoded frames against the vendor's, both lossless, from the same latent.

CHANTIER 2, step one. The vendor-correctness cell found the video family
disagreeing with its vendor at a pinned latent — 17.511 dB, SSIM 0.681, our wood
grain shattered into horizontal streaks where the vendor's is continuous, with a
signature that is DIRECTIONAL (vertical high-frequency energy 1.71x the
vendor's, horizontal 0.87x) and that WORSENS with convergence.

That measurement was taken on frames that had crossed an H.264 encode. The
encode is exonerated by argument — both clips are h264 at the same size and fps,
ours at `-crf 18 -preset medium` against the vendor clip's imageio default, and
ours is the LARGER file — but this chantier will not name a component on an
argument. This tool takes the comparison BEFORE any codec touches either side:

  * ours     — `NBX_DUMP_DECODED_FRAMES` writes what `save_video` computed,
               straight out of the VAE decode;
  * theirs   — `vendor_video_repro.py --frames-dir` writes the pipeline's own
               array, per frame;
  * both     — from the SAME initial latent (`NBX_FIXED_LATENT` on our side,
               `--latents` on theirs), so what is left is the computation.

WHAT IT REPORTS, AND WHY THAT SHAPE
-----------------------------------
Per frame: PSNR, SSIM, and the fraction of spectral energy in the top half of
each axis's frequencies, ours against theirs. The RATIO of those two fractions
is the discriminator this chantier turns on:

    a different sample          raises both axes together
    a horizontal-streak defect  raises the vertical one alone

Reporting per frame also answers a question a single frame cannot: whether the
damage is constant across the clip (a spatial access pattern, the same every
frame) or grows along it (a temporal accumulation). The first points at the VAE
decode's spatial path, the second at its temporal one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_frames(spec: str) -> np.ndarray:
    """Frames from a .npy stack or a directory of PNGs, as uint8 [T, H, W, C]."""
    p = Path(spec)
    if p.is_file() and p.suffix == ".npy":
        return np.load(p)
    if p.is_dir():
        from PIL import Image
        pngs = sorted(p.glob("*.png"))
        if not pngs:
            raise SystemExit(f"no PNG in {p}")
        return np.stack([np.asarray(Image.open(f).convert("RGB")) for f in pngs])
    raise SystemExit(f"not a .npy stack or a directory of PNGs: {spec}")


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def ssim_gray(a: np.ndarray, b: np.ndarray, win: int = 8) -> float:
    """SSIM on luma, uniform window — the same formulation as tools/image_fidelity.py."""
    ga = a.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    gb = b.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    H, W = ga.shape
    h, w = H // win * win, W // win * win
    ga, gb = ga[:h, :w], gb[:h, :w]
    A = ga.reshape(h // win, win, w // win, win).transpose(0, 2, 1, 3).reshape(-1, win * win)
    B = gb.reshape(h // win, win, w // win, win).transpose(0, 2, 1, 3).reshape(-1, win * win)
    ma, mb = A.mean(1), B.mean(1)
    va, vb = A.var(1), B.var(1)
    cov = ((A - ma[:, None]) * (B - mb[:, None])).mean(1)
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))
    return float(s.mean())


def hf_fraction(gray: np.ndarray, axis: int) -> float:
    """Share of spectral energy in the top half of one axis's frequencies."""
    F = np.abs(np.fft.rfft(gray - gray.mean(), axis=axis)) ** 2
    n = F.shape[axis]
    return float(F.take(range(n // 2, n), axis=axis).sum() / F.sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", required=True, help=".npy stack or PNG directory (ours)")
    ap.add_argument("--theirs", required=True, help=".npy stack or PNG directory (vendor)")
    ap.add_argument("--bound-psnr", type=float, default=30.0,
                    help="the repository's bound for 'same computation, different "
                         "implementation' (default 30.0)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ours, theirs = load_frames(a.ours), load_frames(a.theirs)
    if ours.shape != theirs.shape:
        print(f"  frame stacks differ in shape: ours {ours.shape} vs theirs "
              f"{theirs.shape} — that is itself the finding (a frame count or a "
              f"resolution that does not match is not a numeric divergence)")
        if len(ours) != len(theirs):
            return 2
        return 2

    rows = []
    print(f"  {len(ours)} frame(s), {ours.shape[1]}x{ours.shape[2]}, "
          f"lossless on both sides\n")
    print(f"  {'frame':>5s} {'PSNR dB':>9s} {'SSIM':>7s} "
          f"{'HF vert ours':>12s} {'theirs':>8s} {'ratio':>7s} "
          f"{'HF horz ratio':>13s}")
    for i, (o, t) in enumerate(zip(ours, theirs)):
        go = o.astype(np.float64).mean(2)
        gt = t.astype(np.float64).mean(2)
        vo, vt = hf_fraction(go, 0), hf_fraction(gt, 0)
        ho, ht = hf_fraction(go, 1), hf_fraction(gt, 1)
        r = {"frame": i, "psnr_db": round(psnr(o, t), 3),
             "ssim": round(ssim_gray(o, t), 5),
             "hf_vertical_ours": round(vo, 5), "hf_vertical_theirs": round(vt, 5),
             "hf_vertical_ratio": round(vo / vt, 3) if vt else None,
             "hf_horizontal_ratio": round(ho / ht, 3) if ht else None}
        rows.append(r)
        print(f"  {i:5d} {r['psnr_db']:9.3f} {r['ssim']:7.4f} "
              f"{vo:12.5f} {vt:8.5f} {r['hf_vertical_ratio']:7.2f} "
              f"{r['hf_horizontal_ratio']:13.2f}")

    ps = [r["psnr_db"] for r in rows]
    vr = [r["hf_vertical_ratio"] for r in rows if r["hf_vertical_ratio"]]
    hr = [r["hf_horizontal_ratio"] for r in rows if r["hf_horizontal_ratio"]]
    verdict = "AGREES" if min(ps) >= a.bound_psnr else "DIVERGES"
    print(f"\n  PSNR min {min(ps):.3f} / median {float(np.median(ps)):.3f} dB "
          f"against a {a.bound_psnr} dB bound -> {verdict}")
    print(f"  vertical HF ratio   min {min(vr):.2f} max {max(vr):.2f}")
    print(f"  horizontal HF ratio min {min(hr):.2f} max {max(hr):.2f}")
    if max(vr) > 1.25 >= max(hr):
        print("  the excess is VERTICAL only — a horizontal-streak artifact, not a "
              "different sample and not a diffuse numeric drift")
    trend = "constant across the clip" if (max(ps) - min(ps)) < 3.0 else \
            "varies along the clip — look at the temporal path, not only the spatial one"
    print(f"  per-frame PSNR spread {max(ps)-min(ps):.2f} dB: {trend}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"verdict": verdict, "bound_psnr": a.bound_psnr, "frames": rows}, indent=1))
        print(f"  written: {a.out}")
    return 0 if verdict == "AGREES" else 1


if __name__ == "__main__":
    sys.exit(main())
