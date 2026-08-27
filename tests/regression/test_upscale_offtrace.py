"""Permanent off-trace-size upscaler cells — the spatial twin of the
"gates under trace length are blind" lesson dies as a CLASS here.

Every artifact in this file runs `nbx upscale` COLD at an input size
DIFFERENT from its trace size (trace = 64²; cell input = 448²), both
engines, output dimensions verified. Born from
D-RETRACE-SWIN2SR-SYMBOLIC (2026-08-27): the swin2sr graph froze its
window-mask chain at the 64² trace grid and no gate ever ran it at
another size — the spatial-freeze census then showed the WHOLE
Swin-window family shares the class
(validation_outputs/spatial_freeze_census_2026_08_27/VERDICT.md).

Cells:
- real-esrgan-x4: pure conv net, size-agnostic by construction —
  guards the general upscale path at != trace, green from day one.
- swin2SR-classical-sr-x4-64: the known-frozen artifact — xfail with
  the named cause until its two-size re-trace lands, then flips to a
  hard PASS (remove the xfail with the re-trace, never silently).

Supervisor requirement (2026-08-27): the swin2sr re-trace is BORN
with a two-size test — this file is that test's permanent half; the
build-toolchain side validates trace-size byte-identity.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
INPUT_448 = REPO / "benchmarks" / "assets" / "apple_448.png"
SCALE = 4

CELLS = [
    pytest.param("real-esrgan-x4", id="realesrgan-offtrace"),
    pytest.param(
        "swin2SR-classical-sr-x4-64", id="swin2sr-offtrace",
        marks=pytest.mark.xfail(
            reason="D-RETRACE-SWIN2SR-SYMBOLIC: the graph freezes its "
                   "window-mask chain at the 64² trace grid "
                   "(aten.view::18 wall at 448); flips to PASS with "
                   "the two-size re-trace — remove this xfail then",
            strict=False)),
]


@pytest.mark.parametrize("mode_flag", ["compiled", "triton"])
@pytest.mark.parametrize("model", CELLS)
def test_upscale_offtrace_size(model: str, mode_flag: str,
                               tmp_path: Path) -> None:
    from PIL import Image

    from neurobrix.cli.utils import find_model
    try:
        find_model(model)
    except Exception:
        pytest.skip(f"{model} not in the local cache")

    out = tmp_path / f"offtrace_{model}_{mode_flag}.png"
    cmd = [sys.executable, "-u", "-m", "neurobrix", "upscale",
           "--model", model, "--input", str(INPUT_448),
           "--output", str(out)]
    if mode_flag != "compiled":
        cmd += ["--mode", mode_flag]
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"),
           "PYTHONUNBUFFERED": "1"}
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                       env=env, cwd=str(REPO))
    assert r.returncode == 0, (
        f"{model}/{mode_flag} at 448 (trace 64): exit {r.returncode}\n"
        f"... tail:\n{(r.stderr or r.stdout or '')[-600:]}")
    im = Image.open(out)
    assert im.size == (448 * SCALE, 448 * SCALE) and im.mode == "RGB", (
        f"{model}/{mode_flag}: expected {448*SCALE}² RGB, got "
        f"{im.size} {im.mode}")
