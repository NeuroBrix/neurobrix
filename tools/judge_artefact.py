#!/usr/bin/env python3
"""R29 (hardened): judge an artefact, do not trust an agreement between arms.

A line is PROVEN only by an artefact produced from a REAL request and judged by
an instrument EXTERNAL to the engine. Triton-vs-ATen is not external: both are
our stack, and a graph broken upstream breaks them together — the agreement then
says "identical" and proves nothing (real-esrgan: white on BOTH arms for four
days while the tables said "tiling or request").

Three verdicts, never two:
  PROVEN      an artefact exists, passed the degeneracy tests, and its CONTENT
              was judged against the request by something outside the engine
              (a reader for text, an STT for a WAV, an eye for an image).
  MEASURED    the arms agree (PSNR, byte-equality, relative error). Says the
              backends match; says nothing about whether the output is right.
  NOT_MEASURED  it did not run — carries its number (rc, need/available MB).

This module does the mechanical half: existence, geometry, and the degeneracy
tests an image must pass before any eye is worth spending on it. The content
judgment is done by the judge (a human or a model that reads the artefact) and
recorded with the artefact's path, so the page carries links that open.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def image_degeneracy(path: Path, expect_shape=None) -> dict:
    """The tests an image must pass to be worth judging: non-zero spread, more
    than one value, no uniform banding, and the geometry the request implies."""
    import numpy as np
    from PIL import Image

    a = np.asarray(Image.open(path).convert("RGB"))
    std = float(a.std())
    uniq = int(len(np.unique(a)))
    const_rows = int(sum(len(np.unique(a[i])) == 1 for i in range(a.shape[0])))
    const_cols = int(sum(len(np.unique(a[:, j])) == 1 for j in range(a.shape[1])))
    reasons = []
    if std == 0.0:
        reasons.append("zero standard deviation (one flat colour)")
    if uniq <= 1:
        reasons.append("a single distinct value")
    if uniq <= 8:
        reasons.append(f"only {uniq} distinct values")
    if const_rows > a.shape[0] * 0.5:
        reasons.append(f"{const_rows}/{a.shape[0]} rows are a single colour (uniform band)")
    if const_cols > a.shape[1] * 0.5:
        reasons.append(f"{const_cols}/{a.shape[1]} columns are a single colour (uniform band)")
    if expect_shape is not None and tuple(a.shape[:2]) != tuple(expect_shape):
        reasons.append(f"geometry {a.shape[:2]} is not the requested {tuple(expect_shape)}")
    return {"path": str(path), "bytes": path.stat().st_size, "shape": list(a.shape),
            "std": std, "unique_values": uniq,
            "constant_rows": const_rows, "constant_cols": const_cols,
            "degenerate": bool(reasons), "reasons": reasons}


def text_degeneracy(path: Path) -> dict:
    t = path.read_text(errors="replace")
    stripped = t.strip()
    reasons = []
    if not stripped:
        reasons.append("empty output")
    if stripped and len(set(stripped.split())) <= 1:
        reasons.append("a single repeated token")
    return {"path": str(path), "bytes": path.stat().st_size,
            "chars": len(t), "words": len(stripped.split()),
            "degenerate": bool(reasons), "reasons": reasons,
            "head": stripped[:400]}


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    out = []
    for p in argv[1:]:
        path = Path(p)
        if not path.exists():
            out.append({"path": p, "missing": True}); continue
        out.append(image_degeneracy(path) if path.suffix.lower() in (".png", ".jpg", ".jpeg")
                   else text_degeneracy(path))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
