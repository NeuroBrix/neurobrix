"""Prism lists, without executing, the models whose plan differs between hardware profiles
(tools/plan_listing.py) — the regime's one validation per DISTINCT plan (the owner, 2026-10-13 18:05).

What this test does if the code is wrong: the signature cell fails when a change of strategy,
placement, expert store or tiling is not a different plan, or when a planned figure alone makes one;
the classification cell fails when a model with one differing mode is called identical, or a refused
explanation is classed instead of left unknown; the argv cell fails if the explanation does not run
the cell's own request (one definition, `regression_matrix.cell_argv`) or still writes an output; the
end-to-end cell fails if the files do not name the models or a refusal is not said.

Run: PYTHONPATH=src python -m pytest tests/unit/tools/test_prism_lists_the_models_whose_plan_differs_between_profiles.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
import plan_listing as PL  # noqa: E402
import regression_matrix as RM  # noqa: E402


def _plan(strategy="lazy_sequential", store=None, mb=1000, tiling=None):
    return {"strategy": strategy, "loading_mode": "lazy", "planned_memory_mb": mb,
            "components": [{"name": "transformer", "devices": ["cuda:0"], "dtype": "float16", "expert_store": store}],
            "op_level_tiling": tiling, "conv3d_chunks": None, "component_tiling": None, "kv_cache": None}


def test_the_signature_reads_the_plan_not_its_figures():
    assert PL.signature(_plan(mb=1000)) == PL.signature(_plan(mb=9000))
    assert PL.signature(_plan()) != PL.signature(_plan(strategy="layer_streaming"))
    assert PL.signature(_plan()) != PL.signature(_plan(store={"expert_cache_mb": 10}))
    assert PL.signature(_plan()) != PL.signature(_plan(tiling={"x": 1}))


def test_the_classification_needs_every_mode_identical_and_leaves_a_refusal_unknown():
    a, b = PL.signature(_plan()), PL.signature(_plan(strategy="zero3"))
    plans = {"Same": {"native": {"p1": a, "p2": a}, "triton": {"p1": a, "p2": a}},
             "OneModeDiffers": {"native": {"p1": a, "p2": a}, "triton": {"p1": a, "p2": b}},
             "Refused": {"native": {"p1": a, "p2": a}, "triton": {"p1": a, "p2": {"refused": "rc 1: no card"}}}}
    assert PL.classify(plans, ["p1", "p2"]) == {"identical": ["Same"], "differs": ["OneModeDiffers"], "unknown": ["Refused"]}


def test_the_explanation_runs_the_cells_own_request(monkeypatch):
    monkeypatch.setattr(RM, "cell_request", lambda m: ["--prompt", "an apple", "--seed", "42"])
    argv = PL.explain_argv("M", "triton", "default-ff6008b7")
    assert argv[:7] == ["run", "--model", "M", "--prompt", "an apple", "--seed", "42"]
    assert "--triton" in argv and "--output" not in argv
    assert argv[-4:] == ["--hardware", "default-ff6008b7", "--explain-plan", "--json"]


def test_the_explanation_runs_in_the_census_shadow_at_the_profile_not_the_room(tmp_path, monkeypatch):
    import subprocess
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["env"] = cmd, kw["env"]
        return type("R", (), {"returncode": 0, "stdout": "x\n" + json.dumps(_plan()), "stderr": ""})()
    monkeypatch.setattr(subprocess, "run", fake_run)
    plan, why = PL._explain(["run", "--model", "M", "--hardware", "p1", "--explain-plan", "--json"], tmp_path / "src", "py", 10)
    assert plan["strategy"] == "lazy_sequential" and why == ""
    env = seen["env"]
    assert env["CUDA_VISIBLE_DEVICES"] == "" and env["NBX_CENSUS"] == "1" and env["NBX_CENSUS_DEVICES"] == "1"
    assert env["PYTHONPATH"] == str(tmp_path / "src") and "NBX_PRISM_BUDGET_MB" not in env


def test_the_listing_writes_the_three_lists_and_names_a_refusal(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(RM, "cell_request", lambda m: ["--prompt", "x"])
    sigs = {"Same": lambda p: _plan(), "Diff": lambda p: _plan(strategy="zero3" if p == "p2" else "lazy_sequential")}

    def fake_explain(argv, src, python, timeout):
        m = argv[argv.index("--model") + 1]
        p = argv[argv.index("--hardware") + 1]
        if m == "Unknown" and p == "p2":
            return None, "rc 1: Triton on the CPU is refused"
        return (sigs.get(m, lambda p: _plan()))(p), ""
    monkeypatch.setattr(PL, "_explain", fake_explain)
    rc = PL.main(["--models", "Same,Diff,Unknown", "--profiles", "p1,p2", "--src", str(tmp_path / "src"), "--out", str(tmp_path / "out"), "--jobs", "1"])
    assert rc == 0
    o = tmp_path / "out"
    assert (o / "plan_identical.txt").read_text() == "Same\n"
    assert (o / "plan_differs.txt").read_text() == "Diff\n"
    assert (o / "plan_unknown.txt").read_text() == "Unknown\n"
    plans = json.loads((o / "plans.json").read_text())["plans"]
    assert plans["Unknown"]["native"]["p2"] == {"refused": "rc 1: Triton on the CPU is refused"} or plans["Unknown"]["triton"]["p2"]["refused"]
    assert "1 identical, 1 differ, 1 unknown" in capsys.readouterr().out
