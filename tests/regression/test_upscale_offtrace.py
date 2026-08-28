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
