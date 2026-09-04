#!/usr/bin/env python3
"""Image fidelity between two renders — the metric half of the quality gate.

Prints PSNR (dB), SSIM (mean over the image, 8x8 uniform window, gray),
mean and max absolute pixel difference, and the fraction of pixels that
differ by more than 8/255. The eye half of the gate is the R29 inspection
of both PNGs; this tool never replaces it, it makes the eye's verdict
reproducible.

Usage: image_fidelity.py REF.png TEST.png [--json]

Pure numpy + Pillow (file I/O only). Thresholds are NOT decided here —
the caller writes its bound before the measurement, per the locked
protocol.
"""
import argparse
import json
import sys

import numpy as np
from PIL import Image


def _load(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def ssim_gray(a: np.ndarray, b: np.ndarray, win: int = 8) -> float:
    """SSIM on the luma channel with a uniform win x win window
    (Wang et al. 2004 constants, K1=0.01, K2=0.03, L=255)."""
    ga = a @ np.array([0.299, 0.587, 0.114])
    gb = b @ np.array([0.299, 0.587, 0.114])
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    def box(x):
        h, w = x.shape
        hh, ww = h // win * win, w // win * win
        return x[:hh, :ww].reshape(hh // win, win, ww // win, win).mean(axis=(1, 3))

    mu_a, mu_b = box(ga), box(gb)
    saa, sbb, sab = box(ga * ga), box(gb * gb), box(ga * gb)
    va, vb, cov = saa - mu_a ** 2, sbb - mu_b ** 2, sab - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return float(s.mean())


def compare(ref: str, test: str) -> dict:
    a, b = _load(ref), _load(test)
    if a.shape != b.shape:
        raise SystemExit(f"shape mismatch {a.shape} vs {b.shape}")
    d = np.abs(a - b)
    return {
        "ref": ref, "test": test, "shape": list(a.shape),
        "psnr_db": round(psnr(a, b), 3),
        "ssim": round(ssim_gray(a, b), 5),
        "mean_abs_diff": round(float(d.mean()), 4),
        "max_abs_diff": float(d.max()),
        "frac_pixels_diff_gt_8": round(float((d.max(axis=2) > 8).mean()), 5),
        "identical": bool(d.max() == 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    r = compare(args.ref, args.test)
    if args.json:
        print(json.dumps(r))
    else:
        for k, v in r.items():
            print(f"{k:>22}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
