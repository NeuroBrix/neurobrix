"""A gate that holds rows for one memory class says nothing about the other: the computed scope
never calls a cell of a class the gate never judged "without a gate row", and queues no cell of it.

The validation regime (the owner, 2026-10-13 14:20): one class per gate, the card's real VRAM; the
other class only where the plan differs. The first one-class gates (batch C, 2026-10-14 07:59)
were scoped against a 32 GB-only base: the tool found no 16 GB row for any model, marked every model
"no gate row" and queued the whole catalogue in both classes — "0 changed units, 0 reached; 98 + 98
cells". The class a gate holds is read from its rows directories.

What this test does if the code is wrong: the one-class cell fails when a 32 GB-only base marks a
model retraced for its missing 16 GB row (red before this change: every model marked); the
two-class cell fails if a row genuinely missing from a gated class stops being a retrace; the
refusal cell fails if a gate with no rows at all is accepted; the queue cell fails if a reached
cell of a non-gated class survives into the selection.

Run: PYTHONPATH=src python -m pytest tests/unit/tools/test_a_one_class_gate_judges_only_the_class_it_holds.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
import gate_scope as G  # noqa: E402
import regression_matrix as RM  # noqa: E402

MODELS = ("ModelA", "ModelB")


def _gate(tmp_path, classes_with_rows, missing=()):
    cfg = G.load_cfg()
    g = tmp_path / "gate"
    for cls in cfg["classes"]:
        lst = g / cfg["gate"]["lists"].format(cls=cls)
        lst.parent.mkdir(parents=True, exist_ok=True)
        lst.write_text("".join(f"{m} native,triton\n" for m in MODELS))
    (g / cfg["gate"]["timeouts"]).write_text("".join(f"{m} 600\n" for m in MODELS))
    for cls in classes_with_rows:
        rd = g / cfg["gate"]["rows"].format(cls=cls)
        rd.mkdir(parents=True, exist_ok=True)
        rows = [{"model": m, "mode": mo, "rc": 0, "date": "2026-10-14T06:00:00Z", "engine": "deadbeef1234",
                 "gpu": "0", "wall_s": 1.0, "command": "x", "family": "llm"}
                for m in MODELS for mo in ("native", "triton") if (cls, m, mo) not in missing]
        (rd / "rows_card0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return g, cfg


@pytest.fixture
def stubs(monkeypatch):
    # the container index (reads the cache) and the cell argv (reads the family) are not the question here
    monkeypatch.setattr(G.GI, "records", lambda cache, models, index, python, src: {m: {"stamp": [], "gone": False, "ops": [], "arg_records": [], "strings": [], "explicit_dtypes": [], "tensor_dtypes": [], "flow": None, "sigs": {}} for m in models})
    monkeypatch.setattr(RM, "cell_argv", lambda m, mo, out, req=None: ["run", "--model", m])


def test_a_32g_only_base_marks_no_model_retraced_for_its_absent_16g_row(tmp_path, stubs):
    g, cfg = _gate(tmp_path, ["32g"])
    cells = G.Cells(g, cfg, REPO / "src")
    assert cells.gated == {"32g"}
    assert cells.retraced == {}, cells.retraced


def test_a_row_missing_from_a_gated_class_is_still_a_retrace(tmp_path, stubs):
    g, cfg = _gate(tmp_path, ["16g", "32g"], missing={("16g", "ModelB", "triton")})
    cells = G.Cells(g, cfg, REPO / "src")
    assert cells.gated == {"16g", "32g"}
    assert cells.retraced == {"ModelB": "no gate row"}


def test_a_gate_with_no_rows_at_all_is_refused_by_name(tmp_path, stubs):
    g, cfg = _gate(tmp_path, [])
    with pytest.raises(SystemExit, match="holds rows for no class"):
        G.Cells(g, cfg, REPO / "src")


def test_a_reached_cell_of_a_class_the_gate_does_not_hold_is_not_queued(tmp_path, stubs):
    g, cfg = _gate(tmp_path, ["32g"])
    cells = G.Cells(g, cfg, REPO / "src")
    sel = {"16g": {("ModelA", "native"): ["unit x"]}, "32g": {("ModelA", "native"): ["unit x"]}}
    said = []
    G._only_gated(sel, cells, log=said.append)
    assert sel == {"16g": {}, "32g": {("ModelA", "native"): ["unit x"]}}
    assert said and "not gated here" in said[0]
