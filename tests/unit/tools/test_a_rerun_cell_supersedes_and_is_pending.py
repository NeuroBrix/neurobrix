"""A re-run cell supersedes its old row (kept, not deleted) and is pending until judged again.

2026-09-27 02:57 (supervisor): the device-index defect planned every matrix cell on cards 1-3 against board 0's
occupants, so their placement/memory/refusal reds re-run on the fixed main "with the old row marked superseded, not
deleted". Before this the matrix kept every row of a cell side by side and carried the old judgment onto any new row:
a re-run would have been judged by the verdict of the run it replaces. On the old reader these fail.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import regression_matrix as R  # noqa: E402


def _row(date, rc, gpu):
    return {"model": "M", "mode": "native", "gpu": gpu, "rc": rc, "wall_s": 1.0, "date": date, "log": "x"}


def test_the_latest_row_wins_and_the_old_one_rides_along(tmp_path):
    (tmp_path / "rows_card1.jsonl").write_text(json.dumps(_row("2026-09-26T21:00:00Z", 1, "1")) + "\n")
    (tmp_path / "rows_card2.jsonl").write_text(json.dumps(_row("2026-09-27T02:00:00Z", 0, "2")) + "\n")
    (tmp_path / "judgments.jsonl").write_text(json.dumps(
        {"model": "M", "mode": "native", "judged": "old", "verdict": "broken", "date": "2026-09-26 23:30 CEST"}) + "\n")
    rows = R.load_rows(tmp_path)
    assert len(rows) == 1 and rows[0]["rc"] == 0 and rows[0]["superseded"][0]["rc"] == 1
    assert "verdict" not in rows[0], "a verdict taken before the re-run judged the re-run"


def test_a_judgment_after_the_rerun_applies(tmp_path):
    (tmp_path / "rows_card2.jsonl").write_text(json.dumps(_row("2026-09-27T02:00:00Z", 0, "2")) + "\n")
    (tmp_path / "judgments.jsonl").write_text(json.dumps(
        {"model": "M", "mode": "native", "judged": "new", "verdict": "works", "date": "2026-09-27 04:05 CEST"}) + "\n")
    assert R.load_rows(tmp_path)[0]["verdict"] == "works"


def test_a_judgment_stamped_to_the_second_is_read():
    """A judge wrote '2026-09-27 05:58:45 CEST' and every reader of the matrix crashed in
    load_rows (unconverted data remains: :45) — the stamp's precision is the judge's choice."""
    import regression_matrix as R
    a = R._judgment_time({"date": "2026-09-27 05:58:45 CEST"})
    b = R._judgment_time({"date": "2026-09-27 05:58 CEST"})
    assert a - b == 45
