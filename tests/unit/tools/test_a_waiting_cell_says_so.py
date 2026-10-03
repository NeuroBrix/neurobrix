"""A gate cell the host cannot admit SAYS so while it waits, and with --max-wait is refused by name as a row.

The Mac, 2026-10-03 22:40: Voxtral's cell waited in run_cell's loop (need + headroom over the available
memory) with nothing written but host_waiting.json; the pass's last cell waits for ever. Nights were lost
this way: a job "ran" and nothing it produced was written.

What would this file do if the code were wrong? No waiting line -> the first test RED; --max-wait ignored
(the loop never returns) -> the test's own bound trips, RED; the refusal not a row naming why -> RED.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
import regression_matrix as M  # noqa: E402


@pytest.fixture
def never_admitted(monkeypatch, tmp_path):
    clock = {"t": 1000.0}
    monkeypatch.setattr(M, "plan_host_need", lambda *a, **k: (8 << 30, "plan"))
    monkeypatch.setattr(M, "reserve_host", lambda out, need: False)
    monkeypatch.setattr(M, "_mem_available", lambda: 5 << 30)
    monkeypatch.setattr(M.Z, "family_of", lambda m: "llm")
    monkeypatch.setattr(M.time, "time", lambda: clock["t"])

    def sleep(s):
        clock["t"] += s
        if clock["t"] > 1000.0 + 24 * 3600:
            raise AssertionError("the cell waited a day: --max-wait was ignored")
    monkeypatch.setattr(M.time, "sleep", sleep)
    return tmp_path


def test_a_waiting_cell_writes_its_lines_and_is_refused_at_max_wait(never_admitted, capsys):
    out = never_admitted
    row = M.run_cell("Voxtral-Mini-3B-2507", "triton", "0", out, 900, REPO / "src",
                     say_every=300, max_wait=1200)
    lines = [json.loads(l) for l in (out / "waiting.jsonl").read_text().splitlines()]
    assert len(lines) >= 4 and lines[0]["need_gib"] == 8.0 and lines[0]["available_gib"] == 5.0
    assert "WAITING" in capsys.readouterr().out
    assert row["rc"] is None and row["error"].startswith("REFUSED: the host did not admit the cell")
    assert row["model"] == "Voxtral-Mini-3B-2507" and row["mode"] == "triton"


def test_without_max_wait_a_deferrable_cell_still_returns_none(never_admitted):
    assert M.run_cell("Voxtral-Mini-3B-2507", "triton", "0", never_admitted, 900, REPO / "src",
                      wait=False) is None
