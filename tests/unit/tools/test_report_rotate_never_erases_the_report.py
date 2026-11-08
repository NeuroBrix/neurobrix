"""tools/report_rotate.sh keeps the report at most MAX lines without ever erasing it (2026-11-08 04:59: the inline rotation it
replaces, `sed -i "${L}d"` with an empty L, deleted every line of the report when no matching line was left)."""
import os
import subprocess
from pathlib import Path

TOOL = Path(__file__).resolve().parents[3] / "tools" / "report_rotate.sh"


def _rotate(tmp_path, text, max_lines):
    r, o = tmp_path / "r.md", tmp_path / "o.md"
    r.write_text(text)
    env = dict(os.environ, REPORT_BACKUPS=str(tmp_path / "b"), REPORT_MAX_LINES=str(max_lines))
    p = subprocess.run(["bash", str(TOOL), str(r), str(o)], env=env, capture_output=True, text=True)
    return p, r.read_text(), (o.read_text() if o.exists() else "")


def test_a_report_without_dated_lines_is_kept_whole(tmp_path):
    p, report, moved = _rotate(tmp_path, "# title\n- VEILLEUR x\n- RECIPES y\n", 1)
    assert p.returncode == 0 and report == "# title\n- VEILLEUR x\n- RECIPES y\n" and moved == ""


def test_an_empty_report_is_refused(tmp_path):
    p, report, _ = _rotate(tmp_path, "", 1)
    assert p.returncode == 1 and "refused" in p.stderr and report == ""


def test_the_oldest_entry_moves_whole_with_its_continuation(tmp_path):
    p, report, moved = _rotate(tmp_path, "# t\n- 01:00 a\n  a2\n- 01:01 b\n  b2\n", 3)
    assert p.returncode == 0
    assert report == "# t\n- 01:01 b\n  b2\n" and moved == "- 01:00 a\n  a2\n"
    assert len(list((tmp_path / "b").iterdir())) == 1          # the copy taken before the change
