"""`tools/m3_exit_table.py` — M3's exit table: per Triton component, the allocator replay's predicted hold against the
hold the run measured. Its contract: it reads the engine's own lines (the plan's component line, the Triton release
line), keeps a component's dearest load, refuses an empty or mis-globbed input and a command that is not the entry
point by name, and writes only under the workshop's roots.

What would this file do if the tool were wrong? A pattern that no longer matches the engine's line reads nothing and
every figure comes out empty (the parse cells RED); a component released twice keeping its first load (RED); an empty
gate read as an empty, green table (RED); a write outside the atelier accepted (RED)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))

import m3_exit_table as X  # noqa: E402

# the lines as the engine wrote them (PixArt-Sigma-XL-2-1024-MS, nbxtensor 2a0fc723, 2026-10-29)
PLAN = """\
host memory     1455 MB on the triton engine = resident 219 MB + engine 207 MB + held 0 MB (eager)
  text_encoder         -> cuda:0  weights 9083 MB + activations 78 MB + overhead 0 MB; the allocator's replay holds 11094 MB at its peak (the pool x1.352 over the walk)  (peak at aten.mul::7)
  vae                  -> cuda:0  weights 160 MB + activations 1920 MB + overhead 0 MB; the allocator's replay holds 4105 MB at its peak (the pool x1.180 over the walk)  (peak at aten.native_group_norm::23)
  transformer          -> cuda:0  weights 1165 MB + activations 628 MB + overhead 0 MB  (peak at aten.mm::3)
drain cliff     cuda:0: the Triton hot loop drains its dead tensors at 1907 MB queued
"""
LOG = """\
   [Triton] 'text_encoder': released 221 weights, held at most 10157 MB from the driver since its load (0 MB of it held before); the device still holds 1061 MB live
   [Triton] 'transformer': released 604 weights, held at most 8081 MB from the driver since its load (1071 MB of it held before); the device still holds 2998 MB live
   [Triton] 'vae': released 60 weights, held at most 3000 MB from the driver since its load; the device still holds 0 MB live
   [Triton] 'vae': released 60 weights, held at most 3500 MB from the driver since its load (200 MB of it held before); the device still holds 0 MB live
"""


def test_the_plan_line_is_read_per_component():
    p = X.predicted(PLAN)
    assert p["text_encoder"] == {"device": "cuda:0", "weights_mb": 9083, "activations_mb": 78,
                                 "replay_held_mb": 11094, "pool_ratio": 1.352}
    assert p["vae"]["replay_held_mb"] == 4105
    assert p["transformer"]["replay_held_mb"] is None and p["transformer"]["weights_mb"] == 1165   # a tree without
    assert set(p) == {"text_encoder", "vae", "transformer"}                                      # the replay says so


def test_the_release_line_is_read_and_a_component_keeps_its_dearest_load():
    m = X.measured(LOG)
    assert m["text_encoder"] == {"held": 10157, "before": 0}
    assert m["transformer"] == {"held": 8081, "before": 1071}
    assert m["vae"] == {"held": 3500, "before": 200}            # own 3300 beats the first load's 3000


def test_a_row_compares_the_replay_and_the_arena_with_the_own_hold(monkeypatch):
    monkeypatch.setattr(X, "arena_factor", lambda hw, dev: 1.3)
    rows = {r["component"]: r for r in X.table_rows({"model": "M", "mode": "triton"}, PLAN, 0, LOG, "hw")}
    t = rows["transformer"]
    assert t["measured_own_mb"] == 8081 - 1071 and t["arena_mb"] == round(1165 + 628 * 1.3)
    assert t["replay_over_measured"] is None                    # no replay stated, no ratio invented
    e = rows["text_encoder"]
    assert e["replay_over_measured"] == round(11094 / 10157, 3)
    assert e["arena_over_measured"] == round(round(9083 + 78 * 1.3) / 10157, 3)


def test_an_empty_or_misglobbed_gate_is_refused_by_name(tmp_path):
    with pytest.raises(X.ExitTableRefused, match="no gate rows"):
        X.gate_rows([str(tmp_path)])                            # a directory with no rows_card*.jsonl
    with pytest.raises(X.ExitTableRefused, match="no gate rows"):
        X.gate_rows([str(tmp_path / "rows_card9.jsonl")])
    f = tmp_path / "rows_card0.jsonl"
    f.write_text(json.dumps({"model": "M", "mode": "compiled", "command": "x"}) + "\n")
    with pytest.raises(X.ExitTableRefused, match="no Triton row"):
        X.gate_rows([str(tmp_path)])


def test_a_replayed_cell_is_read_at_its_last_row(tmp_path):
    f = tmp_path / "rows_card0.jsonl"
    f.write_text("".join(json.dumps({"model": "M", "mode": "triton", "command": c}) + "\n" for c in ("first", "last")))
    assert [r["command"] for r in X.gate_rows([str(tmp_path)])] == ["last"]


def test_only_an_entry_point_command_is_asked_for_its_plan():
    argv = X.explain_argv("/py -m neurobrix run --model M --triton --certified-only --output /x.png")
    assert argv[-1] == "--explain-plan" and argv[:4] == ["/py", "-m", "neurobrix", "run"]
    with pytest.raises(X.ExitTableRefused, match="not an entry-point command"):
        X.explain_argv("/py tools/house_script.py --model M")


def test_a_table_outside_the_workshop_is_refused(tmp_path):
    import atelier
    f = tmp_path / "rows_card0.jsonl"
    f.write_text(json.dumps({"model": "M", "mode": "triton", "command": "/py -m neurobrix run --model M"}) + "\n")
    with pytest.raises(atelier.OutsideTheAtelier):
        X.main(["--rows", str(tmp_path), "--tree", str(REPO), "--out", "/var/tmp/exit_table.tsv"])
