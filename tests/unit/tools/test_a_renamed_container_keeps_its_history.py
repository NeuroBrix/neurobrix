"""A renamed container keeps its history: a record written under its former name is joined under its
current one, and a record written under the current name wins over the former.

The owner's rule of 2026-09-28 03:06 renamed five containers (a publication-format suffix is not a
model's name). Dated records — regression rows, judgments, last proofs, campaign cells — are never
rewritten; the tools read their names through `tools/container_renames.py`. Without it a re-run under
the new name is a second model beside the old one, and every renamed model looks never proven: with
the rename table emptied, each test here fails.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import container_renames as N  # noqa: E402
import regression_matrix as R  # noqa: E402

OLD, NEW = "Wan2.1-T2V-1.3B-Diffusers", "Wan2.1-T2V-1.3B"


def _row(model, date, rc):
    return {"model": model, "mode": "native", "gpu": "2", "rc": rc, "wall_s": 1.0, "date": date, "log": "x"}


def test_a_record_under_the_former_name_is_keyed_by_the_current_one():
    assert N.by_current_name({OLD: "proof of 09-13", "TinyLlama-1.1B-Chat-v1.0": "t"}) == {
        NEW: "proof of 09-13", "TinyLlama-1.1B-Chat-v1.0": "t"}


def test_the_current_name_wins_over_the_former_one():
    assert N.by_current_name({OLD: "before the rename", NEW: "after it"}) == {NEW: "after it"}
    assert N.by_current_name({NEW: "after it", OLD: "before the rename"}) == {NEW: "after it"}


def test_a_rerun_under_the_new_name_supersedes_the_row_under_the_old_one(tmp_path):
    (tmp_path / "rows_card2.jsonl").write_text(
        json.dumps(_row(OLD, "2026-09-27T21:00:00Z", 1)) + "\n"
        + json.dumps(_row(NEW, "2026-09-28T09:00:00Z", 0)) + "\n")
    (tmp_path / "judgments.jsonl").write_text(json.dumps(
        {"model": OLD, "mode": "native", "judged": "old", "verdict": "broken", "date": "2026-09-27 23:30 CEST"}) + "\n")
    rows = R.load_rows(tmp_path)
    assert len(rows) == 1, "one container, one cell — not a second model beside the old one"
    assert rows[0]["model"] == NEW and rows[0]["rc"] == 0
    assert rows[0]["superseded"][0]["recorded_as"] == OLD
    assert "verdict" not in rows[0], "the judgment of the run before the re-run judged the re-run"
