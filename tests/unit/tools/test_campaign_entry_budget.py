"""A cell whose known cost exceeds the campaign's budget is refused at entry.

On 2026-09-10 the certified-directory proof spent about fifteen hours of a quiet
rig on `Allegro`: arm A killed at 28 800 s, arm B killed at 24 958 s, both
`rc < 0`, verdict `ran: false`. The cost was not a surprise — the same cell had
already been killed at the same wall on an earlier flight, and the projection
from its own log was ~31 h per arm against an 8 h timeout. The next flight
excluded Allegro by hand, so the lesson was learnt, but only after the spend.

There is already a guard INSIDE the cell: an arm that produces no output ends
the container, because the byte gate needs every arm and no remaining arm can
change the verdict. What was missing is the guard at the DOOR — the one that
never lets the first arm start.

The rule: a cell whose estimated cost exceeds the campaign's budget is refused
before it consumes a second of card, loudly, with its number.

Run: PYTHONPATH=src python -m pytest tests/unit/tools/test_campaign_entry_budget.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[3] / "tools" / "precision_zoo_campaign.py"


def _tool():
    spec = importlib.util.spec_from_file_location("precision_zoo_campaign", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _cell(tmp_path: Path, name: str, payload) -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    if payload is not None:
        (d / "result.json").write_text(json.dumps(payload))
    return d


def test_a_cell_never_measured_has_no_estimate(tmp_path):
    """No record, no number — the guard must not invent one and must not
    refuse a cell it knows nothing about."""
    m = _tool()
    assert m.cell_cost_estimate(_cell(tmp_path, "unknown", None), timeout=28800) is None


def test_a_killed_arm_estimates_at_least_the_timeout_times_the_arms(tmp_path):
    """The live Allegro record: both arms killed at the wall."""
    m = _tool()
    d = _cell(tmp_path, "Allegro", {
        "A": {"rc": -9, "wall_s": 28800.238490581512},
        "B": {"rc": -15, "wall_s": 24958.0},
        "gate": {"ran": False}})
    est, basis = m.cell_cost_estimate(d, timeout=28800)
    assert est >= 2 * 28800, f"a killed arm bounds the cost from below, got {est}"
    assert "killed" in basis.lower() or "tue" in basis.lower()
    assert "28800" in basis


def test_a_measured_cell_estimates_from_its_arms(tmp_path):
    m = _tool()
    d = _cell(tmp_path, "Qwen", {
        "A": {"rc": 0, "wall_s": 100.0},
        "B": {"rc": 0, "wall_s": 140.0},
        "gate": {"ran": True}})
    est, basis = m.cell_cost_estimate(d, timeout=28800)
    assert 200 <= est <= 300, f"expected roughly the two arms, got {est}"
    assert "140" in basis or "100" in basis


def test_the_guard_refuses_a_cell_that_cannot_fit(tmp_path):
    """Allegro against an eight-hour campaign budget."""
    m = _tool()
    d = _cell(tmp_path, "Allegro", {
        "A": {"rc": -9, "wall_s": 28800.0},
        "B": {"rc": -15, "wall_s": 24958.0}})
    verdict = m.budget_refusal(d, budget_s=8 * 3600, timeout=28800)
    assert verdict is not None, "a 16 h cell must be refused by an 8 h budget"
    assert "8" in verdict or "28800" in verdict or "57600" in verdict, (
        f"the refusal must carry its number: {verdict}")


def test_the_guard_admits_a_cell_that_fits(tmp_path):
    m = _tool()
    d = _cell(tmp_path, "Qwen", {
        "A": {"rc": 0, "wall_s": 100.0},
        "B": {"rc": 0, "wall_s": 140.0}})
    assert m.budget_refusal(d, budget_s=8 * 3600, timeout=28800) is None


def test_no_budget_declared_refuses_nothing(tmp_path):
    """The guard is opt-in: a campaign that declares no budget keeps today's
    behaviour exactly."""
    m = _tool()
    d = _cell(tmp_path, "Allegro", {"A": {"rc": -9, "wall_s": 28800.0}})
    assert m.budget_refusal(d, budget_s=None, timeout=28800) is None
