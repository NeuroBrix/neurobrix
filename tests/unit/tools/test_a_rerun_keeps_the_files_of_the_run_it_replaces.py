"""A cell run again keeps the files of the run it replaces (`regression_matrix.keep_superseded`).

The row a rerun supersedes is kept (the supervisor's rule of 2026-09-27 02:57), but its log and artefact were
overwritten by the new run: the final gate's Wan2.2-I2V native 32g (2026-11-04) took another plan under host contention
(another sha, a 0 MB peak) and its rerun alone erased the log that held why. What this test does if the code were wrong:
the old log overwritten or deleted — red at the first assertion; the new row not citing where they went — red at the
second; a second move onto the same place silently replacing the first — red at the third."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import regression_matrix as R  # noqa: E402


def test_the_replaced_runs_log_and_artefact_are_moved_aside_and_cited(tmp_path):
    d = tmp_path / "Wan2.2-I2V-A14B"; d.mkdir()
    (d / "native.log").write_text("the plan that ran under contention\n")
    (d / "native.mp4").write_bytes(b"old video")
    (d / "triton.log").write_text("the other branch, untouched\n")
    kept = R.keep_superseded(d, "native")
    assert sorted(Path(k).read_bytes() for k in kept) == [b"old video", b"the plan that ran under contention\n"]
    assert not (d / "native.log").exists() and not (d / "native.mp4").exists() and (d / "triton.log").exists()
    assert all(Path(k).parent.parent == d / "superseded" for k in kept)
    os.utime(d / "triton.log", (0, 0))
    (d / "superseded" / "19700101T000000Z").mkdir()
    (d / "superseded" / "19700101T000000Z" / "triton.log").write_text("earlier")
    with pytest.raises(FileExistsError, match="would overwrite"):
        R.keep_superseded(d, "triton")


def test_a_first_run_moves_nothing(tmp_path):
    assert R.keep_superseded(tmp_path, "native") == []


def test_the_cell_keeps_them_before_it_runs_and_its_row_cites_them():
    body = Path(R.__file__).read_text()
    body = body[body.index("def _run_cell("):]
    body = body[:body.index("\ndef ")]
    assert body.index("keep_superseded(d, mode)") < body.index("Z.run(cmd")
    assert 'row["superseded_files"] = kept' in body and "art.unlink()" not in body
