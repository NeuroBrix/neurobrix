"""A cell SAYS at once what it runs at — and the only thing it ever waits for is the gate's PAUSE door.

The Mac, 2026-10-03 22:34-22:36: Voxtral's cell waited for host admission with run.log unchanged and no
`neurobrix run` process; only host_waiting.json moved, and a GPU night can be lost that way without a trace.
The answer then was a "waits for host admission ... need, available, headroom" line every WAIT_SAY_S.

The owner's rule of 2026-10-12 22:45 (commit c8ddcfd36): a correctness cell never waits for host memory and is
never refused for it. The admission loop and its waiting line are gone from `run_cell` and exist nowhere else,
so the two old tests (which asserted that line) are replaced by what the cell says and does now: one line at
once, "runs at the plan, host figure N GiB (source), no admission", even on a host with less available memory
than the plan states, without asking the ledger and without sleeping; and the one door it still waits behind,
`<out>/PAUSE` (nothing runs beside a gate).

What each test would do if the code were wrong: a run_cell admitting against the host again asks the trapped
ledger and never prints the line — the first test RED (seen: c8ddcfd36's run_cell hunk reverted); a run_cell
that dropped the PAUSE door runs the cell while the gate holds it — the second test RED (seen: the door's
`while` loop removed by injection).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import regression_matrix as R  # noqa: E402

GiB = 1 << 30


def _host_below_the_plan(monkeypatch, sleep):
    def trap(*a, **k):
        raise AssertionError("run_cell asked the host ledger: a correctness cell is never admitted against host memory")
    ran = []
    monkeypatch.setattr(R, "reserve_host", trap)
    monkeypatch.setattr(R, "release_host", trap)
    monkeypatch.setattr(R, "plan_host_need", lambda *a, **k: (6 * GiB, "plan"))
    monkeypatch.setattr(R, "_run_cell", lambda *a, **k: ran.append(1) or {"rc": 0})
    monkeypatch.setattr(R, "_host_bytes", lambda: 24 * GiB)
    monkeypatch.setattr(R, "_mem_available", lambda: 5 * GiB)       # less than the plan's 6 GiB
    monkeypatch.setattr(R.time, "sleep", sleep)
    return ran


def test_a_cell_says_at_once_it_runs_at_the_plan_without_admission(tmp_path, monkeypatch, capsys):
    def sleep(s):
        raise AssertionError("the cell slept with no PAUSE: it waited for host memory")
    ran = _host_below_the_plan(monkeypatch, sleep)
    row = R.run_cell("m", "triton", "0", tmp_path, 60, tmp_path)
    said = capsys.readouterr().out.splitlines()
    assert row["rc"] == 0 and ran == [1]
    assert "[matrix] m triton: runs at the plan, host figure 6 GiB (plan), no admission" in said, said
    assert not [l for l in said if "waits for host admission" in l], said
    assert row["host_reserved"] == 6 * GiB and row["host_reserved_from"] == "plan"


def test_the_only_door_a_cell_waits_behind_is_the_gates_pause(tmp_path, monkeypatch):
    (tmp_path / "PAUSE").write_text("")
    slept = []

    def sleep(s):                       # the gate ends after three polls and lifts its door
        assert not ran, "the cell ran while the gate's PAUSE was up"
        slept.append(s)
        if len(slept) == 3:
            (tmp_path / "PAUSE").unlink()
    ran = _host_below_the_plan(monkeypatch, sleep)
    row = R.run_cell("m", "triton", "0", tmp_path, 60, tmp_path)
    assert len(slept) == 3 and ran == [1] and row["rc"] == 0
