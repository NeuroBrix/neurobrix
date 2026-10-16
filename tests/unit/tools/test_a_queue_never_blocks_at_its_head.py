"""A run's queue never blocks at its head on memory — and since 2026-10-12 it never consults host memory at all.

VALIDATE 49 waited from 15:38 to 17:23 (2026-10-04) on CogVideoX-2b (need 9.1 GiB, 4.0 GiB available, the
owner's browser holding the rest): the chain handed one cell per call and the card did nothing for almost two
hours (the supervisor's 17:23: a queue that blocks at its head on memory is a tool defect). The answer then was
to reorder the queue by host memory (the cells that fit first, smallest first), defer a cell that never fits
(`--defer-after`) and plan a waiting cell again against the memory free now.

The owner's rule of 2026-10-12 22:45 (commit c8ddcfd36) removed the cause instead: a correctness cell runs at
Prism's plan and is never admitted against host memory, so `run_cell` never returns "not now" and the queue
has nothing to wait on. The three old tests asserted the reordering, the deferral and the re-planning; that
behaviour no longer exists in the matrix nor anywhere else (`reserve_host` has no caller in tools/ or src/), so
they are replaced by the rule itself on the same host (4 GiB available, A needing 9): every cell runs, in the
order asked, without a skip, a deferral, a re-plan or a single sleep, and with the ledger made a trap.

What each test would do if the code were wrong: a run_cell admitting against the host again asks the ledger —
every test RED on the trap (seen: c8ddcfd36's run_cell hunk reverted); a queue that still deferred or reordered
by memory would break the order or the rc (RED).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import regression_matrix as R  # noqa: E402

GB = 1 << 30
NEEDS = {"A": 9 * GB, "B": 3 * GB, "C": 1 * GB}


def _host(tmp_path, monkeypatch):
    """A host with 4 GiB available, less than A's plan states; the ledger is a trap, sleeping is counted."""
    for m in NEEDS:
        (tmp_path / "cache" / m).mkdir(parents=True)
        (tmp_path / "cache" / m / "manifest.json").write_text("{}")
    monkeypatch.setattr(R, "CACHE", tmp_path / "cache")
    seen = {"ran": [], "sleeps": 0, "planned": []}

    def sleep(s):
        seen["sleeps"] += 1
        raise AssertionError("the run slept: a correctness cell never waits for host memory")

    def trap(*a, **k):
        raise AssertionError("the host ledger was asked: a correctness cell is never admitted against host memory")

    def plan(model, mode, gpu, src):
        seen["planned"].append(model)
        return NEEDS[model], "plan-unified (test)"

    monkeypatch.setattr(R.time, "sleep", sleep)
    monkeypatch.setattr(R, "_mem_available", lambda: 4 * GB)
    monkeypatch.setattr(R, "_host_bytes", lambda: 24 * GB)
    monkeypatch.setattr(R, "plan_host_need", plan)
    monkeypatch.setattr(R, "reserve_host", trap)
    monkeypatch.setattr(R, "release_host", trap)

    def run(model, mode, gpu, out, timeout, src):
        seen["ran"].append(model)
        return {"model": model, "mode": mode, "rc": 0, "wall_s": 1.0, "date": "2026-10-04T17:30:00"}

    monkeypatch.setattr(R, "_run_cell", run)
    return seen


def _args(tmp_path, **kw):
    a = dict(models="A,B,C", modes="triton", gpu="0", out=str(tmp_path / "out"), src=str(tmp_path / "src"),
             timeout=60, rerun=False, gate_lists=None, allow_certified_dir_override=False, defer_after=None)
    a.update(kw)
    return argparse.Namespace(**a)


def test_every_cell_runs_in_the_order_asked_without_a_skip(tmp_path, monkeypatch, capsys):
    seen = _host(tmp_path, monkeypatch)
    assert R.cmd_run(_args(tmp_path)) == 0
    assert seen["ran"] == ["A", "B", "C"], f"run order {seen['ran']}: the queue is never reordered by host memory"
    said = capsys.readouterr().out
    assert "skipped" not in said and "none fits" not in said, said
    assert "A triton: runs at the plan, host figure 9 GiB (plan-unified (test)), no admission" in said, said


def test_defer_after_never_leaves_a_correctness_cell_without_a_row(tmp_path, monkeypatch, capsys):
    seen = _host(tmp_path, monkeypatch)
    rc = R.cmd_run(_args(tmp_path, defer_after=600))
    assert seen["ran"] == ["A", "B", "C"]
    assert rc == 0, "no cell is deferred for host memory"
    assert "DEFERRED" not in capsys.readouterr().out


def test_without_defer_after_a_run_still_gives_every_cell_its_row(tmp_path, monkeypatch):
    """A gate needs every row: each asked cell has one, read back from the files."""
    seen = _host(tmp_path, monkeypatch)
    assert R.cmd_run(_args(tmp_path)) == 0
    rows = {r["model"] for f in (tmp_path / "out").glob("rows_card*.jsonl") for r in R.read_jsonl(f)}
    assert rows == {"A", "B", "C"} and sorted(seen["ran"]) == ["A", "B", "C"]


def test_a_cell_is_planned_once_and_never_again_for_memory(tmp_path, monkeypatch, capsys):
    """Re-planning a waiting cell against the memory free now (2026-10-04) existed only because cells waited;
    a cell that never waits is planned once, by run_cell, and its row carries that plan's figure."""
    seen = _host(tmp_path, monkeypatch)
    assert R.cmd_run(_args(tmp_path, models="A")) == 0
    assert seen["planned"] == ["A"], f"plans asked: {seen['planned']}"
    assert "planned again" not in capsys.readouterr().out
    row = next(r for r in R.read_jsonl(tmp_path / "out" / "rows_card0.jsonl"))
    assert row["host_reserved"] == 9 * GB and row["host_reserved_from"] == "plan-unified (test)"
