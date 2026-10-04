"""A judge that raises gives its row an error; it never kills the harness run.

mochi-1-preview on the Mac (2026-10-04 17:58): the cell ran to its end and saved its video, then
`video_degeneracy` raised ModuleNotFoundError (imageio absent from the served environment). The
exception left `run_cell` with no row and the whole run died; the next model ran only because a
chain called the harness again. A judge that cannot judge is a named state of the row, never a
pass and never the end of the run.
"""
from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(TOOLS))

import regression_matrix as R  # noqa: E402


def _raising(*a, **k):
    raise ModuleNotFoundError("No module named 'imageio'")


def test_a_raising_judge_names_the_row_and_returns(tmp_path, monkeypatch):
    art = tmp_path / "out.mp4"
    art.write_bytes(b"\0" * 64)
    monkeypatch.setattr(R, "video_degeneracy", _raising)
    m = R.mechanical(art, "video", (64, 160))
    assert m["judge_error"] == "ModuleNotFoundError: No module named 'imageio'"
    assert m["degenerate"] is None          # not judged: neither degenerate nor clean


def test_the_table_shows_the_judge_error_never_a_ran_cell():
    rows = [{"model": "m", "mode": "triton", "family": "video", "rc": 0, "wall_s": 1,
             "mechanical": {"judge_error": "ModuleNotFoundError: No module named 'imageio'", "degenerate": None}}]
    cell = R.table_cell(rows[0])
    assert cell.startswith("JUDGE ERROR ModuleNotFoundError") and "ran " not in cell


def test_the_harness_declares_the_judges_video_reader():
    import re                               # tomllib is 3.11+; the rack's engine python is 3.10
    text = (TOOLS.parent / "pyproject.toml").read_text()
    dev = re.search(r"^dev = \[(.*?)\]", text, re.M | re.S).group(1)
    names = [re.split(r"[<>=!~ ]", d.strip().strip('"'))[0] for d in dev.split(",") if d.strip()]
    assert "imageio" in names, names
