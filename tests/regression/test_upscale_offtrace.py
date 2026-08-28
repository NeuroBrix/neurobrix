"""Permanent TWO-SIZE upscaler cells — the spatial twin of the
"gates under trace length are blind" lesson dies as a CLASS here.

Every artifact runs `nbx upscale` COLD at TWO input sizes, on both
engines, with output dimensions verified:

  * its OWN trace size, read from the installed container (so the
    cell follows any future stimulus change instead of hardcoding
    one) — this half catches regressions introduced BY a re-trace;
  * an OFF-TRACE size — the half that catches everything a
    trace-sized gate is structurally blind to.

Both halves are load-bearing. When the swin2sr recovery config was
first attached under `flow.stages`, it silently redefined the
model's flow and dropped a component: the off-trace cell merely
stayed red, while the TRACE-SIZE cell flipped green->red and made
the regression attributable in one cycle (2026-08-28).

Born from the upscaler trace-value collision + frozen window-count
pair (DETTE: D-RETRACE-SWIN2SR-SYMBOLIC; the class covers the seven
Swin-window artifacts — see
validation_outputs/spatial_freeze_census_2026_08_27/VERDICT.md).

Runnable: python3 -m pytest tests/regression/test_upscale_offtrace.py -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CACHE = Path.home() / ".neurobrix" / "cache"
OFF_TRACE_HW = (448, 448)  # the committed bench asset's size

CELLS = [
    pytest.param("real-esrgan-x4", id="realesrgan"),
    # Hard PASS since 2026-08-28: the collision-free stimulus plus the
    # window-count recovery closed D-RETRACE-SWIN2SR-SYMBOLIC, and this
    # cell XPASSed on all four combinations. The xfail was removed with
    # the re-trace, never left to rot into a silent pass.
    pytest.param("swin2SR-classical-sr-x4-64", id="swin2sr"),
]


def _trace_hw(model: str) -> tuple[int, int]:
    """The artifact's OWN trace spatial size, read from its container."""
    for graph in sorted((CACHE / model / "components").glob("*/graph.json")):
        tensors = (json.loads(graph.read_text()) or {}).get("tensors", {})
        for tid, info in tensors.items():
            shape = info.get("shape") or []
            if tid.startswith("input::") and len(shape) == 4:
                return int(shape[2]), int(shape[3])
    raise AssertionError(f"{model}: no 4-D graph input found in the container")


def _upscale_factor(model: str) -> int:
    """Scale factor from the container manifest / model name."""
    manifest = json.loads((CACHE / model / "manifest.json").read_text())
    for key in ("upscale", "scale"):
        if key in manifest:
            return int(manifest[key])
    # Fall back to the trailing xN in the artifact name (hub convention).
    for token in model.replace("-", " ").replace("_", " ").split():
        if token.lower().startswith("x") and token[1:].isdigit():
            return int(token[1:])
    raise AssertionError(f"{model}: cannot determine the upscale factor")


def _make_input(hw: tuple[int, int], dest: Path) -> Path:
    from PIL import Image

    height, width = hw
    img = Image.new("RGB", (width, height))
    img.putdata([(x * 4 % 256, y * 4 % 256, (x + y) * 2 % 256)
                 for y in range(height) for x in range(width)])
    img.save(dest)
    return dest


@pytest.mark.parametrize("mode", ["compiled", "triton"])
@pytest.mark.parametrize("size", ["trace", "offtrace"])
@pytest.mark.parametrize("model", CELLS)
def test_upscale_two_sizes(model: str, size: str, mode: str,
                           tmp_path: Path) -> None:
    from PIL import Image

    from neurobrix.cli.utils import find_model
    try:
        find_model(model)
    except Exception:
        pytest.skip(f"{model} not in the local cache")

    hw = _trace_hw(model) if size == "trace" else OFF_TRACE_HW
    if size == "offtrace" and hw == _trace_hw(model):
        pytest.skip("off-trace size coincides with the trace size — "
                    "pick a different OFF_TRACE_HW")
    scale = _upscale_factor(model)

    src = _make_input(hw, tmp_path / f"in_{model}_{size}.png")
    out = tmp_path / f"out_{model}_{size}_{mode}.png"
    cmd = [sys.executable, "-u", "-m", "neurobrix", "upscale",
           "--model", model, "--input", str(src), "--output", str(out)]
    if mode != "compiled":
        cmd += ["--mode", mode]
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"),
           "PYTHONUNBUFFERED": "1"}
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                       env=env, cwd=str(REPO))
    assert r.returncode == 0, (
        f"{model}/{mode} at {hw[0]}x{hw[1]} ({size}): exit "
        f"{r.returncode}\n... tail:\n{(r.stderr or r.stdout or '')[-600:]}")

    expected = (hw[1] * scale, hw[0] * scale)  # PIL size is (W, H)
    im = Image.open(out)
    assert im.size == expected and im.mode == "RGB", (
        f"{model}/{mode} ({size}): expected {expected} RGB, got "
        f"{im.size} {im.mode}")


# ---------------------------------------------------------------------------
# TONE FIDELITY — the value twin of the shape cells above.
#
# The cells above assert output DIMENSIONS. That is exactly the hole the
# swin2sr mean-addback defect walked through: it shipped to the hub with
# correct dimensions, correct sharpness, correct 4x factor, and every
# pixel shifted by the model's RGB mean (grey background instead of
# white). A shape gate is structurally blind to it.
#
# These cells assert VALUES, on the real bench asset (the synthetic
# gradient used above has no uniform background, so it could not carry
# this check even with a value assertion):
#
#   1. input-referenced (universal, no fixture needed) — super-resolution
#      preserves low-frequency content, so the output mean must track the
#      input mean, and a uniform background must survive. This half works
#      for ANY upscaler, including ones with no banked vendor arm.
#   2. vendor-referenced (when a vendor arm is banked) — the output mean
#      must match the vendor stack's own output.
#
# Thresholds live in data/upscaler_tone_reference.json, measured across
# four independent vendor stacks — never inline constants.
# ---------------------------------------------------------------------------

TONE_REF = json.loads(
    (Path(__file__).parent / "data" / "upscaler_tone_reference.json").read_text())


def _tone_stats(path: Path, patch: int) -> tuple[float, list[float]]:
    """Global mean and the background triplet (median of corner patches)."""
    import numpy as np
    from PIL import Image

    a = np.asarray(Image.open(path).convert("RGB")).astype(float)
    corners = np.concatenate([
        a[:patch, :patch].reshape(-1, 3), a[:patch, -patch:].reshape(-1, 3),
        a[-patch:, :patch].reshape(-1, 3), a[-patch:, -patch:].reshape(-1, 3)])
    return float(a.mean()), [float(v) for v in np.median(corners, axis=0)]


def _tone_params() -> list:
    """One cell per (artifact, engine).

    An artifact carrying a KNOWN defect is marked xfail(strict=True) —
    named, so it reports as a tracked defect rather than as suite noise,
    and strict, so the marker cannot rot into a silent pass: the day the
    re-trace lands, the cell XPASSes and the suite goes red until the
    marker is removed. Same discipline that retired the swin2sr shape
    xfail on 2026-08-28.
    """
    out = []
    for model, cfg in TONE_REF["artifacts"].items():
        defect = cfg.get("known_defect")
        for engine in cfg["engines"]:
            marks = ([pytest.mark.xfail(reason=defect, strict=True)]
                     if defect else [])
            out.append(pytest.param(model, engine, marks=marks,
                                    id=f"{model}-{engine}"))
    return out


@pytest.mark.parametrize("model,mode", _tone_params())
def test_upscale_tone_fidelity(model: str, mode: str, tmp_path: Path) -> None:
    from neurobrix.cli.utils import find_model
    try:
        find_model(model)
    except Exception:
        pytest.skip(f"{model} not in the local cache")

    tol = TONE_REF["tolerances"]
    patch = TONE_REF["background_probe"]["corner_patch_px"]
    asset = REPO / TONE_REF["input_asset"]
    if not asset.exists():
        pytest.skip(f"bench asset missing: {asset}")

    # Run at the artifact's OWN trace size. One cell, one defect class:
    # the off-trace freeze is the two-size cells' job above, and letting
    # it crash this cell would hide the tonal verdict behind an exit 1
    # (observed 2026-08-28 on swinir / hat / swin2sr-x2 / realworld).
    from PIL import Image

    hw = _trace_hw(model)
    src = tmp_path / f"tone_in_{model}.png"
    Image.open(asset).convert("RGB").resize(
        (hw[1], hw[0]), Image.LANCZOS).save(src)

    out = tmp_path / f"tone_{model}_{mode}.png"
    cmd = [sys.executable, "-u", "-m", "neurobrix", "upscale",
           "--model", model, "--input", str(src), "--output", str(out)]
    if mode != "compiled":
        cmd += ["--mode", mode]
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"),
           "PYTHONUNBUFFERED": "1"}
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                       env=env, cwd=str(REPO))
    assert r.returncode == 0, (
        f"{model}/{mode}: exit {r.returncode}\n"
        f"... tail:\n{(r.stderr or r.stdout or '')[-600:]}")

    in_mean, in_bg = _tone_stats(src, patch)
    out_mean, out_bg = _tone_stats(out, patch)

    # 1. Input-referenced — catches a global tonal shift with no fixture.
    d_mean = abs(out_mean - in_mean)
    assert d_mean <= tol["mean_abs_delta_vs_input"], (
        f"{model}/{mode}: output mean {out_mean:.2f} vs input {in_mean:.2f} "
        f"(delta {d_mean:.2f} > {tol['mean_abs_delta_vs_input']}). An "
        f"upscaler preserves low-frequency content; this size of shift is a "
        f"tonal defect, not resampling. Check whether the traced unit "
        f"dropped a parent-forward normalisation.")

    d_bg = [abs(o - i) for o, i in zip(out_bg, in_bg)]
    assert max(d_bg) <= tol["background_abs_delta_per_channel"], (
        f"{model}/{mode}: background {out_bg} vs input {in_bg} "
        f"(per-channel delta {[round(v, 1) for v in d_bg]} > "
        f"{tol['background_abs_delta_per_channel']}). A uniform background "
        f"must survive upscaling.")

    # 2. Vendor-referenced — only where a vendor arm is banked AND this
    #    cell ran at the size that arm was measured at. Comparing a
    #    trace-size run against a 448-measured vendor mean would be
    #    comparing two different inputs.
    vendor = (TONE_REF["artifacts"][model] or {}).get("vendor")
    if vendor and list(hw) == TONE_REF.get("vendor_measured_hw"):
        d_vendor = abs(out_mean - vendor["mean"])
        assert d_vendor <= tol["mean_abs_delta_vs_vendor"], (
            f"{model}/{mode}: output mean {out_mean:.2f} vs vendor "
            f"{vendor['mean']:.2f} (delta {d_vendor:.2f} > "
            f"{tol['mean_abs_delta_vs_vendor']}). Vendor reference: "
            f"{vendor['source']}")
