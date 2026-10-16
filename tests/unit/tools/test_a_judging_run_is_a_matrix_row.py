"""A run at the request a model's quality is judged at is a matrix row like every other.

2026-10-16: the Wan judging runs (Wan2.2-I2V-A14B at 40 steps, Wan2.1-I2V-14B-480P at 50, Wan2.1-VACE-1.3B at 30, at the
registry's confirmation values) were made with `neurobrix run` beside the matrix, because `regression_matrix run` makes
only the gate request. Judged from outside, they had no row: artefact-judgments.jsonl and benchmarks.md, built from rows,
never counted them. `run --request judging` makes the derived (judging) request, writes the kind in the row, is refused
in a gate, and never shares a matrix directory with gate rows (a cell's row is keyed by model and mode).
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import regression_matrix as R  # noqa: E402


@pytest.fixture
def kinds(monkeypatch):
    monkeypatch.setattr(R, "gate_request", lambda m, f: ["--steps", "4"])
    monkeypatch.setattr(R, "derived_request", lambda m, f: ["--steps", "40"])
    monkeypatch.setattr(R.Z, "family_of", lambda m: "video")
    monkeypatch.setattr(R, "REQUEST_KIND", "gate")


def test_a_judging_run_makes_the_judging_request_and_a_gate_run_the_gate_one(kinds, monkeypatch):
    assert R.cell_request("m") == ["--steps", "4"]
    assert "--steps" in R.cell_argv("m", "native", "x.mp4") and "4" in R.cell_argv("m", "native", "x.mp4")
    monkeypatch.setattr(R, "REQUEST_KIND", "judging")
    assert R.cell_request("m") == ["--steps", "40"]
    argv = R.cell_argv("m", "native", "x.mp4")
    assert argv[argv.index("--steps") + 1] == "40"


def _cache(tmp_path, monkeypatch):
    (tmp_path / "cache" / "a").mkdir(parents=True)
    (tmp_path / "cache" / "a" / "manifest.json").write_text("{}")
    monkeypatch.setattr(R, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(R, "run_cell", lambda *a, **k: pytest.fail("a cell ran before the refusal"))


def test_a_gate_at_the_judging_request_is_refused_before_any_cell(tmp_path, monkeypatch):
    _cache(tmp_path, monkeypatch)
    (tmp_path / "l").write_text("a native,triton\n")
    args = NS(models="a", modes="native,triton", gpu="0", out=str(tmp_path / "o"), src=str(tmp_path), timeout=60,
              rerun=False, gate_lists=[str(tmp_path / "l")], known_reds=str(tmp_path / "reds.yml"), request="judging")
    assert R.cmd_run(args) == 2
    assert not (tmp_path / "o").exists()


@pytest.mark.parametrize("held,asked", [({}, "judging"), ({"request_kind": "judging"}, "gate")])
def test_a_judging_run_never_shares_a_directory_with_gate_rows(tmp_path, monkeypatch, held, asked):
    _cache(tmp_path, monkeypatch)
    out = tmp_path / "matrix_x"
    out.mkdir()
    # a row written before the field existed is a gate row
    (out / "rows_card2.jsonl").write_text(json.dumps({"model": "a", "mode": "native", "rc": 0, **held}) + "\n")
    args = NS(models="a", modes="native", gpu="3", out=str(out), src=str(tmp_path), timeout=60, rerun=False,
              gate_lists=None, request=asked)
    assert R.cmd_run(args) == 2
    assert R.REQUEST_KIND == "gate"
