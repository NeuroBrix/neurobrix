"""A re-trace whose POINT is a different stimulus cannot be judged by a diff that assumes one.

The re-trace gate's vocabulary — witnessed, symbolized, re-expressed — compares a shape argument
against the SAME trace: it reads a literal becoming an expression by checking that the expression's
trace value EQUALS the old literal. When the repair is the stimulus, that equality never holds.

real-esrgan-x2, 2026-09-16: its pixel-unshuffle went from the literal 32 (64//2 at a 64x64 trace,
where height, width and the first convolution's 64 channels are one number) to `floordiv(s1, 2)` of
trace 56 (112//2 at the collision-free 112x80). The repair is exactly what the gate exists to
recognise, and the gate scored it **0 symbolized, 2193 changes beyond annotation, FAIL** — because
every recorded shape in the graph moved with the stimulus.

A count that cannot be interpreted may not read as a refusal. It is named instead, with what the
bytes did, and the artefacts decide. What still refuses, refuses: a corrupted dim, a routing field
removed or changed, a run that failed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import retrace_zoo as R  # noqa: E402


def _ctx(**names):
    return {"symbols": {f"s{i}": {"name": k, "trace_value": v}
                        for i, (k, v) in enumerate(names.items())}}


def test_the_same_stimulus_is_not_a_change():
    same = _ctx(batch=1, height=112, width=80)
    assert R.stimulus_change(same, _ctx(batch=1, height=112, width=80)) is None


def test_the_upscaler_case_is_detected_and_says_what_moved():
    old = _ctx(batch=1, height=64, width=64)
    new = _ctx(batch=1, height=112, width=80)
    moved = R.stimulus_change(old, new)
    assert moved == {"height": {"old": [64], "new": [112]},
                     "width": {"old": [64], "new": [80]}}, moved
    assert "batch" not in moved, "a dimension whose trace did not move is not part of the change"


def test_a_dimension_only_one_side_declares_is_not_a_moved_trace():
    """A symbol the new graph gained is a different finding — the diff's own business, not this."""
    assert R.stimulus_change(_ctx(height=112), _ctx(height=112, time=10)) is None


def test_empty_or_missing_contexts_are_not_a_change():
    for a, b in ((None, None), ({}, {}), (None, _ctx(height=112)), (_ctx(height=112), None)):
        assert R.stimulus_change(a, b) is None
