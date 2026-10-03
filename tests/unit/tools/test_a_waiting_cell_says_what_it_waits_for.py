"""A cell the host cannot admit yet SAYS that it waits, what it needs and what the host has — at once and
then every WAIT_SAY_S — instead of sleeping in `run_cell`'s admission loop without a line.

The Mac, 2026-10-03 22:34-22:36: Voxtral's cell waited for admission (its need plus the headroom above the
11.5 GB available) with run.log unchanged and no `neurobrix run` process; only host_waiting.json moved, and a
GPU night can be lost that way without a trace. Before this branch the first test fails: nothing is printed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import regression_matrix as R  # noqa: E402

GiB = 1 << 30


def _waits_then_admits(monkeypatch, refusals):
    calls = {"n": 0}

    def reserve(out, need):
        calls["n"] += 1
        return calls["n"] > refusals
    monkeypatch.setattr(R, "reserve_host", reserve)
    monkeypatch.setattr(R, "release_host", lambda out: None)
    monkeypatch.setattr(R, "plan_host_need", lambda *a, **k: (6 * GiB, "plan"))
    monkeypatch.setattr(R, "_run_cell", lambda *a, **k: {"rc": 0})
    monkeypatch.setattr(R, "_host_bytes", lambda: 24 * GiB)
    monkeypatch.setattr(R, "_mem_available", lambda: 5 * GiB)
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    return calls


def test_a_waiting_cell_prints_its_need_and_the_available_memory(tmp_path, monkeypatch, capsys):
    _waits_then_admits(monkeypatch, refusals=3)
    monkeypatch.setattr(R, "WAIT_SAY_S", 0)
    row = R.run_cell("m", "triton", "0", tmp_path, 60, tmp_path)
    said = [l for l in capsys.readouterr().out.splitlines() if "waits for host admission" in l]
    assert row["rc"] == 0
    assert said, "a waiting cell printed nothing"
    assert "need 6.0 GiB" in said[0] and "available 5.0 GiB" in said[0] and "headroom 1.5 GiB" in said[0]


def test_the_waiting_line_is_bounded_by_its_period(tmp_path, monkeypatch, capsys):
    _waits_then_admits(monkeypatch, refusals=50)
    monkeypatch.setattr(R, "WAIT_SAY_S", 10 ** 9)          # a period no test reaches: only the first refusal speaks
    R.run_cell("m", "triton", "0", tmp_path, 60, tmp_path)
    said = [l for l in capsys.readouterr().out.splitlines() if "waits for host admission" in l]
    assert len(said) == 1
