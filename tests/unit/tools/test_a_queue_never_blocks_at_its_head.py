"""A run's queue never blocks at its head on memory: the cells that fit now go first, smallest first, and the
big one is retried each time; a run given `--defer-after` leaves a cell that never fits without a row, named.

VALIDATE 49 waited from 15:38 to 17:23 (2026-10-04) on CogVideoX-2b (need 9.1 GiB, 4.0 GiB available, the
owner's browser holding the rest): the chain hands one cell per call, a pass's LAST cell waits for the budget
without end, and the card did nothing for almost two hours (the supervisor's 17:23: a queue that blocks at its
head on memory is a tool defect). Before this branch: the cells run in list order after a skip (B before C),
and a cell that never fits keeps the run waiting (the fake clock runs out).
"""
import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import regression_matrix as R  # noqa: E402

GB = 1 << 30
NEEDS = {"A": 9 * GB, "B": 3 * GB, "C": 1 * GB}


class _OutOfTime(Exception):
    pass


def _host(tmp_path, monkeypatch, frees_after=None):
    """A host with 4 GiB available; after `frees_after` sleeps of the harness, 16 GiB."""
    for m in NEEDS:
        (tmp_path / "cache" / m).mkdir(parents=True)
        (tmp_path / "cache" / m / "manifest.json").write_text("{}")
    monkeypatch.setattr(R, "CACHE", tmp_path / "cache")
    clock = {"t": 1_000_000.0, "sleeps": 0}
    ran = []

    def sleep(s):
        clock["sleeps"] += 1
        clock["t"] += s
        if clock["sleeps"] > 2000:
            raise _OutOfTime("the run is still waiting after 2 000 sleeps")

    def available():
        return 16 * GB if frees_after is not None and clock["sleeps"] >= frees_after else 4 * GB

    monkeypatch.setattr(R.time, "sleep", sleep)
    monkeypatch.setattr(R.time, "time", lambda: clock["t"])
    monkeypatch.setattr(R, "_mem_available", available)
    monkeypatch.setattr(R, "_host_bytes", lambda: 24 * GB)
    monkeypatch.setattr(R, "plan_host_need", lambda model, mode, gpu, src: (NEEDS[model], "plan-unified (test)"))
    monkeypatch.setattr(R, "reserve_host", lambda out, need, headroom=None: need <= available())
    monkeypatch.setattr(R, "release_host", lambda out: None)

    def run(model, mode, gpu, out, timeout, src):
        ran.append(model)
        return {"model": model, "mode": mode, "rc": 0, "wall_s": 1.0, "date": "2026-10-04T17:30:00"}

    monkeypatch.setattr(R, "_run_cell", run)
    return ran


def _args(tmp_path, **kw):
    a = dict(models="A,B,C", modes="triton", gpu="0", out=str(tmp_path / "out"), src=str(tmp_path / "src"),
             timeout=60, rerun=False, gate_lists=None, allow_certified_dir_override=False, defer_after=None)
    a.update(kw)
    return argparse.Namespace(**a)


def test_the_cells_that_fit_go_first_smallest_first_and_the_head_is_retried(tmp_path, monkeypatch, capsys):
    ran = _host(tmp_path, monkeypatch, frees_after=3)
    assert R.cmd_run(_args(tmp_path)) == 0
    assert ran == ["C", "B", "A"], f"run order {ran}: after the head is skipped the smallest cell goes first"
    said = capsys.readouterr().out
    assert "A triton: skipped" in said and "retr" in said, said


def test_a_cell_that_never_fits_is_deferred_by_name_and_the_rest_run(tmp_path, monkeypatch, capsys):
    ran = _host(tmp_path, monkeypatch)
    rc = R.cmd_run(_args(tmp_path, defer_after=600))
    assert ran == ["C", "B"]
    assert rc == R.DEFERRED_RC
    captured = capsys.readouterr()
    assert "A triton: DEFERRED" in captured.out
    assert "have no row" not in captured.err, "a deferred cell is named as deferred, not as a lost row"


def test_without_defer_after_a_run_still_waits_for_its_cell(tmp_path, monkeypatch):
    """A gate keeps waiting: every asked cell must get a row; deferring is the caller's choice."""
    ran = _host(tmp_path, monkeypatch, frees_after=50)
    assert R.cmd_run(_args(tmp_path)) == 0
    assert sorted(ran) == ["A", "B", "C"]


def test_a_waiting_cell_is_planned_again_against_the_memory_free_now(tmp_path, monkeypatch, capsys):
    """Prism plans against the memory free when it is asked: DeepSeek-Coder-V2-Lite planned 17 137 MB at 19:30
    (2026-10-04) with 17 144 MB available, memory then fell to 15 GiB, and the run waited 30 min on that one
    figure until it was deferred. A plan asked again at 15 GiB is a lower rung's. Before this test: the need
    is asked once, A never fits, the fake clock runs out."""
    ran = _host(tmp_path, monkeypatch)
    asked = []

    def plan(model, mode, gpu, src):     # A's plan at the memory free now: its top rung needs 9, the next one 3
        asked.append(model)
        if model != "A":
            return NEEDS[model], "plan-unified (test)"
        return (9 * GB if len([m for m in asked if m == "A"]) <= 2 else 3 * GB), "plan-unified (test)"   # the first skip asks twice

    monkeypatch.setattr(R, "plan_host_need", plan)
    assert R.cmd_run(_args(tmp_path, models="A")) == 0
    assert ran == ["A"]
    said = capsys.readouterr().out
    assert "A triton: planned again" in said and "9.0 GiB -> 3.0 GiB" in said, said
