"""A cell whose sweep arm swept nothing has not measured — it refuses.

On 2026-09-10 the re-anchored proof's first cell, `deepseek-moe-16b-chat`,
produced `speedup: 1.0288` — A 34,34 s against B 33,38 s — and sat in the table
as if it meant something. Its control arm had swept **zero** keys: it took all
eight from the replay cache its own first repetition had written, and paid none
of the cost it exists to pay. The cell compared "certified" to "certified", and
the answer was one.

That is the same class as a gate that never passed and a runner that counted
segments instead of cells: **a measurement must prove it took place.** A ratio of
one produced by an arm that did no work is worse than a missing row, because a
missing row is visibly missing.

The predicate is the arm's own telemetry, not a guess: the lever arm reports how
many keys it was served from the certified directory, how many it swept at
runtime, and how many it read back from the replay cache. An arm that served
nothing AND swept nothing exercised nothing.

Run: PYTHONPATH=src python -m pytest tests/unit/tools/test_campaign_lever_must_measure.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[3] / "tools" / "precision_zoo_campaign.py"


def _tool():
    spec = importlib.util.spec_from_file_location("precision_zoo_campaign", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_live_contaminated_cell_is_refused():
    """deepseek-moe-16b-chat as recorded on 2026-09-10 16:37, verbatim."""
    m = _tool()
    reason = m.vacuous_lever_reason({
        "lever": "NBX_AUTOTUNE_CERTIFIED", "paired": 3,
        "A": {"certified_served": 8, "swept": 0, "from_replay": 0, "exec_s": 34.34},
        "B": {"certified_served": 0, "swept": 0, "from_replay": 8, "exec_s": 33.38},
    })
    assert reason is not None, "an arm that swept nothing must not sit in the table"
    assert "0" in reason
    assert "replay" in reason.lower() or "rejeu" in reason.lower(), (
        f"the refusal must name the cause it can see: {reason}")


def test_a_cell_whose_control_swept_is_admitted():
    """The same model this morning, at paired=1: B really swept its eight."""
    m = _tool()
    assert m.vacuous_lever_reason({
        "lever": "NBX_AUTOTUNE_CERTIFIED", "paired": 1,
        "A": {"certified_served": 8, "swept": 0, "from_replay": 0, "exec_s": 66.85},
        "B": {"certified_served": 0, "swept": 8, "from_replay": 0, "exec_s": 76.99},
    }) is None


def test_a_model_that_touches_no_autotuned_kernel_is_refused_too():
    """Nothing served, nothing swept, nothing replayed anywhere: the cell has
    no opinion about the certified directory and must not pretend to one."""
    m = _tool()
    reason = m.vacuous_lever_reason({
        "lever": "NBX_AUTOTUNE_CERTIFIED", "paired": 1,
        "A": {"certified_served": 0, "swept": 0, "from_replay": 0},
        "B": {"certified_served": 0, "swept": 0, "from_replay": 0},
    })
    assert reason is not None


def test_a_cell_with_no_lever_is_not_judged():
    """The guard belongs to the A/B lever protocol; a row that declares no
    lever is none of its business."""
    m = _tool()
    assert m.vacuous_lever_reason({
        "A": {"certified_served": 0, "swept": 0}, "B": {}}) is None


def test_an_arm_with_no_telemetry_is_not_judged():
    """A run that never reached the autotune recap (a crash, an unsupported
    path) is a failure the gate already reports — not a vacuous measurement."""
    m = _tool()
    assert m.vacuous_lever_reason({
        "lever": "NBX_AUTOTUNE_CERTIFIED",
        "A": {"certified_served": None, "swept": None},
        "B": {"certified_served": None, "swept": None},
    }) is None
