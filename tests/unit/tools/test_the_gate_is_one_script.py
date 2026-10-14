"""The gate is ONE committed script (the validation regime, Hocine 2026-10-14 14:20 / 18:05): the computed scope,
one class, two scheduler instances (the any-card instance recording under the gate's class), the tally reading the
committed records — judgments included.

What this test does if the code is wrong: the any-card cell fails when the scheduler records a cell under the card's
class instead of the gate's (`--out-class`); the class cell fails when a plain instance stops recording under the
card's class; the split cell fails when a plan-identical model is placed on the class's cards or another model on
any card; the refusal cell fails when a missing plan-identical list is taken as "no model identical"; the judgments
cells fail when the latest judgment does not win, when an input without a judgment is accepted, when a second
collect rewrites an unchanged record, or when an unjudged wanted pair is not named; the tally cell fails when the
gate's tally does not name the wanted pairs no judgment covers or counts them as reds.

Run: python -m pytest tests/unit/tools/test_the_gate_is_one_script.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

import gate_run  # noqa: E402
import gate_sched  # noqa: E402
import gate_tally  # noqa: E402
import judgments  # noqa: E402


def _args(tmp_path, **over):
    base = dict(gate2=str(tmp_path / "gate2"), gate=str(tmp_path / "gate"), tree=str(ROOT), run_md=str(tmp_path / "RUN.md"),
                holds=str(tmp_path / "holds"), class_split_mib=24576, heavy_share=0.25, hardware="rig", poll=30.0, reap=5.0,
                need_workers=1, peak_rows=[], adopt=[], state=str(tmp_path / "state"), out_class=None, label="gate2",
                python=sys.executable)
    base.update(over)
    return gate_sched.build_parser().parse_args(sum(([f"--{k.replace('_', '-')}", str(v)] for k, v in base.items()
                                                      if v is not None and not isinstance(v, list)), []))


def test_the_any_card_instance_records_a_cell_under_the_gates_class(tmp_path):
    gate_sched.configure(_args(tmp_path, out_class="32g"))
    cmd = gate_sched.gate_cmd(0, "16g", "Janus-Pro-7B native,triton 7200")
    assert gate_sched.OUT("16g") == "32g"
    assert str(tmp_path / "gate" / "matrix_32g") in cmd
    assert str(tmp_path / "gate2" / "gate_32g.list") in cmd
    assert cmd[cmd.index("--known-reds") + 1].endswith("known-cause-reds.32g.yml")
    assert "--gpu" in cmd and cmd[cmd.index("--gpu") + 1] == "0", "the row names the card it ran on"


def test_a_plain_instance_records_under_the_cards_class(tmp_path):
    gate_sched.configure(_args(tmp_path))
    cmd = gate_sched.gate_cmd(2, "32g", "Janus-Pro-7B native,triton 7200")
    assert gate_sched.OUT("16g") == "16g"
    assert str(tmp_path / "gate" / "matrix_32g") in cmd and str(tmp_path / "gate2" / "gate_32g.list") in cmd


def test_the_split_sends_plan_identical_models_to_any_card_and_the_others_to_the_class():
    own, anyc = gate_run.split_queue(["# a comment", "", "Janus-Pro-7B native,triton 7200", "Allegro native,triton 7200 rerun"],
                                     {"Janus-Pro-7B"})
    assert own == ["Allegro native,triton 7200 rerun"] and anyc == ["Janus-Pro-7B native,triton 7200"]


def test_a_missing_plan_identical_list_is_refused_by_name(tmp_path):
    with pytest.raises(gate_run.Refused, match="plan-identical list is not there"):
        gate_run.read_list(tmp_path / "plan-identical.txt")


def _matrix(tmp_path, name, cls, rows=(), judged=()):
    d = tmp_path / name / f"matrix_{cls}"
    d.mkdir(parents=True)
    if rows:
        (d / "rows_card0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if judged:
        (d / "judgments.jsonl").write_text("".join(json.dumps(j) + "\n" for j in judged))
    return d


def test_the_judgments_record_keeps_the_latest_per_model_and_mode_and_refuses_an_empty_input(tmp_path):
    j1 = {"model": "A", "mode": "native", "judged": "looked at the PNG: an apple", "verdict": "works", "date": "2026-10-12 18:55 CEST"}
    j2 = {"model": "A", "mode": "native", "judged": "looked again: a smear", "verdict": "broken", "date": "2026-10-13 09:00 CEST"}
    j3 = {"model": "A", "mode": "triton", "judged": "looked at the PNG: an apple", "verdict": "works", "date": "2026-10-12 18:56 CEST"}
    _matrix(tmp_path, "old", "32g", judged=[j1, j3])
    _matrix(tmp_path, "new", "32g", judged=[j2])
    rec = tmp_path / "docs" / "judgments.jsonl"
    assert judgments.main(["collect", "--record", str(rec), str(tmp_path / "old"), str(tmp_path / "new")]) == 0
    rows = judgments.read_jsonl(rec)
    assert [(r["model"], r["mode"], r["verdict"]) for r in rows] == [("A", "native", "broken"), ("A", "triton", "works")]
    assert rows[0]["class"] == "32g" and rows[0]["source"].endswith("new/matrix_32g")
    before = rec.read_text()
    assert judgments.main(["collect", "--record", str(rec), str(tmp_path / "old"), str(tmp_path / "new")]) == 0
    assert rec.read_text() == before, "a second run over the same inputs changes nothing"
    (tmp_path / "empty").mkdir()
    assert judgments.main(["collect", "--record", str(rec), str(tmp_path / "empty")]) == 2
    assert rec.read_text() == before, "a refused input writes nothing"


def test_the_check_names_the_wanted_pairs_without_a_judgment(tmp_path, capsys):
    rec = tmp_path / "judgments.jsonl"
    rec.write_text(json.dumps({"model": "A", "mode": "native", "class": "32g", "verdict": "works", "judged": "seen",
                               "date": "2026-10-12 18:55 CEST", "source": "x"}) + "\n")
    gl = tmp_path / "gate_32g.list"
    gl.write_text("A native,triton\nB native,triton\n")
    assert judgments.main(["check", "--record", str(rec), "--gate-list", str(gl)]) == 1
    out = capsys.readouterr().out
    assert "1 of 4 wanted (model, mode) judged; 3 not judged" in out
    assert "NOT JUDGED A:triton" in out and "NOT JUDGED B:native" in out and "NOT JUDGED B:triton" in out
    assert judgments.main(["check", "--record", str(tmp_path / "none.jsonl"), "--gate-list", str(gl)]) == 2


def test_the_tally_names_the_unjudged_pairs_without_counting_them_red(tmp_path, capsys):
    gate = tmp_path / "gate"
    (gate / "gate2").mkdir(parents=True)
    (gate / "gate2" / "gate_32g.list").write_text("A native,triton\n")
    row = lambda m, mode, rc: {"model": m, "mode": mode, "rc": rc, "wall_s": 10, "date": "2026-10-14T10:00:00Z"}  # noqa: E731
    _matrix(tmp_path, "gate", "32g", rows=[row("A", "native", 0), row("A", "triton", 0)])
    _matrix(tmp_path, "base", "32g", rows=[row("A", "native", 0), row("A", "triton", 0)])
    tree = tmp_path / "tree" / "docs" / "reference"
    tree.mkdir(parents=True)
    (tree / "known-cause-reds.32g.yml").write_text("reds: []\n")
    (tree / "judgments.jsonl").write_text(json.dumps({"model": "A", "mode": "native", "class": "32g", "verdict": "works",
                                                       "judged": "seen", "date": "2026-10-12 18:55 CEST", "source": "x"}) + "\n")
    rc = gate_tally.main(["--gate", str(gate), "--base", str(tmp_path / "base"), "--tree", str(tmp_path / "tree"), "--classes", "32g"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "2 green / 0 red" in out and "judged 1 / not judged 1" in out and "NOT JUDGED A:triton" in out


def test_a_base_gate_without_its_timeouts_is_refused_before_any_scope(tmp_path):
    B = tmp_path / "base"
    (B / "gate2").mkdir(parents=True)
    (B / "gate2" / "gate_32g.list").write_text("A native,triton\n")
    (B / "matrix_32g").mkdir()
    ident = tmp_path / "plan-identical.txt"
    ident.write_text("A\n")
    a = gate_run.build_parser().parse_args([
        "--campaign", str(tmp_path / "c"), "--tree", str(ROOT), "--base-gate", str(B), "--plan-identical", str(ident),
        "--run-md", str(tmp_path / "RUN.md"), "--holds", str(tmp_path / "h"), "--hardware", "rig",
        "--class-split-mib", "24576", "--heavy-share", "0.25", "--need-workers", "1", "--plan"])
    with pytest.raises(gate_run.Refused, match="timeouts"):
        gate_run.steps(a)


def test_the_any_card_instance_serves_the_other_class_and_reads_the_shared_holds(tmp_path):
    assert gate_run.any_card_classes("32g") == ["16g"] and gate_run.any_card_classes("16g") == ["32g"]
    with pytest.raises(gate_run.Refused, match="not one the scheduler knows"):
        gate_run.any_card_classes("24g")
    B = tmp_path / "base"
    (B / "gate2").mkdir(parents=True)
    (B / "gate2" / "gate_32g.list").write_text("A native,triton\n")
    (B / "matrix_32g").mkdir()
    (B / "timeouts.txt").write_text("A 600\n")
    ident = tmp_path / "plan-identical.txt"
    ident.write_text("A\n")
    a = gate_run.build_parser().parse_args([
        "--campaign", str(tmp_path / "c"), "--tree", str(ROOT), "--base-gate", str(B), "--plan-identical", str(ident),
        "--run-md", str(tmp_path / "RUN.md"), "--holds", str(tmp_path / "shared_holds"), "--hardware", "rig",
        "--class-split-mib", "24576", "--heavy-share", "0.25", "--need-workers", "1", "--plan"])
    for name, cmd, *_ in gate_run.steps(a):
        if name in ("sched", "sched_any"):
            assert cmd[cmd.index("--holds") + 1] == str(tmp_path / "shared_holds"), f"{name} reads private holds"


def test_the_carry_count_reads_the_record(tmp_path):
    (tmp_path / "carries.yml").write_text("carries:\n- model: A\n  mode: native\n- model: B\n  mode: triton\n")
    assert gate_run.carried(tmp_path) == 2 and gate_run.carried(tmp_path / "none") == 0
