"""Serve WARM path gate — the class response to the warm-boundary lesson.

The serving warm path (InferenceEngine.generate — the same code the
serve daemon dispatches to) is a separate input/output surface from
the CLI cold path, and gaps there have repeatedly slept through cold
gates (4th recurrence 2026-08-26: the upscaler family fell through
to the video input contract and broke on serve warm, BOTH engines,
while every cold cell was green — S3_FINDING_SERVE_UPSCALER_WARM).

Contract of this gate: at least one row per family exercises the
warm path in every battery, with the OUTPUT SHAPE AND TYPE verified,
on both engines, under short budgets (a budget-killed cell reads as
"slow" when it is broken — the Flex lesson).

Currently covered: upscaler (the family whose warm break motivated
the gate). Extend one small row per family as warm coverage grows —
never delete a family from here once added.

Runnable: python3 -m pytest tests/regression/test_serve_warm.py -q
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

UPSCALER_MODEL = "real-esrgan-x4"
UPSCALE_FACTOR = 4
INPUT_SIZE = 64  # synthetic input — self-contained, no asset debt


def _synthetic_input(tmp_path: Path) -> Path:
    """A small deterministic RGB gradient — content is irrelevant to
    the gate (shape/type verified, not pixels)."""
    from PIL import Image

    img = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE))
    img.putdata([(x * 4 % 256, y * 4 % 256, (x + y) * 2 % 256)
                 for y in range(INPUT_SIZE) for x in range(INPUT_SIZE)])
    p = tmp_path / "warm_input.png"
    img.save(p)
    return p


@pytest.mark.parametrize("mode", ["compiled", "triton"])
def test_upscaler_serve_warm_output_shape(mode: str, tmp_path: Path) -> None:
    from PIL import Image

    from neurobrix.core.prism.autodetect import get_or_create_default_profile
    from neurobrix.serving.engine import InferenceEngine

    try:
        from neurobrix.cli.utils import find_model
        find_model(UPSCALER_MODEL)
    except Exception:
        pytest.skip(f"{UPSCALER_MODEL} not in the local cache")

    engine = InferenceEngine(UPSCALER_MODEL,
                             get_or_create_default_profile(), mode=mode)
    engine.load()
    try:
        input_png = _synthetic_input(tmp_path)
        expected = INPUT_SIZE * UPSCALE_FACTOR

        for request_idx in (1, 2):  # request 2 = the warm-reuse property
            result = engine.generate(image_path=str(input_png))
            assert "outputs" in result, (
                f"warm request {request_idx} ({mode}): no outputs in "
                f"result — keys {sorted(result)}")
            out_path = tmp_path / f"warm_out_{mode}_{request_idx}.png"
            saved = engine.save_output(result["outputs"], str(out_path))
            im = Image.open(saved)
            assert im.size == (expected, expected) and im.mode == "RGB", (
                f"warm request {request_idx} ({mode}): expected "
                f"{expected}x{expected} RGB, got {im.size} {im.mode}")
    finally:
        engine.unload()
