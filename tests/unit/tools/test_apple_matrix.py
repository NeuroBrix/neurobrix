"""The matrix cell classifier decides verified / cause / not-measured, and a
shape disagreement is a CAUSE, never a soft pass.

The classifier is the core the 47x3 matrix rests on; a cell that mis-scores is
a wrong row in the table Hocine reads. Both directions on each verdict.

Runnable: PYTHONPATH=src python3 -m pytest tests/unit/tools/test_apple_matrix.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from apple_matrix import (                                          # noqa: E402
    Verdict, classify_image, classify_text, classify_failure, not_measured,
    MODES, REFERENCE_MODE)


def test_the_three_modes_and_the_reference():
    assert MODES == ("compiled", "triton", "triton-sequential")
    assert REFERENCE_MODE == "compiled"


def test_a_matching_image_is_verified():
    rng = np.random.default_rng(0)
    ref = (rng.random((896, 896, 3)) * 255).astype(np.uint8)
    arm = ref.copy()
    arm[0, 0, 0] = min(255, int(arm[0, 0, 0]) + 1)      # one ULP of noise
    cell = classify_image(ref, arm)
    assert cell.verdict is Verdict.VERIFIED
    assert cell.number > 40


def test_a_shape_disagreement_is_a_cause_not_a_pass():
    """real-esrgan: 128x128 where 896x896 is owed. Must NOT read as close."""
    ref = np.zeros((896, 896, 3), dtype=np.uint8)
    arm = np.zeros((128, 128, 3), dtype=np.uint8)
    cell = classify_image(ref, arm)
    assert cell.verdict is Verdict.CAUSE
    assert "shape" in cell.detail


def test_a_visibly_different_image_is_a_cause():
    rng = np.random.default_rng(1)
    ref = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    arm = (rng.random((64, 64, 3)) * 255).astype(np.uint8)   # unrelated
    cell = classify_image(ref, arm)
    assert cell.verdict is Verdict.CAUSE
    assert cell.number < 40


def test_byte_identical_text_is_verified():
    assert classify_text("the fox", " the fox ").verdict is Verdict.VERIFIED


def test_diverging_text_is_a_cause():
    assert classify_text("the fox", "the dog").verdict is Verdict.CAUSE


def test_a_failure_cell_carries_origin_and_ir_verdict():
    cell = classify_failure("2-D reduce > 1024", origin="amont", ir_verdict="non")
    assert cell.verdict is Verdict.CAUSE
    assert "amont" in cell.detail and "non" in cell.detail


def test_not_measured_is_never_blank():
    cell = not_measured("artefact 23.6 GB > memory budget")
    assert cell.verdict is Verdict.NOT_MEASURED
    assert cell.detail.strip(), "a not-measured cell must carry its number"
