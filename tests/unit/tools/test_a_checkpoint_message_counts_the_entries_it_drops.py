"""A checkpoint's commit headline counts the entries it DROPS, not only those it adds and changes.

`file_certification` drops every entry of another bits epoch from the file it files into. On the
recertification of 2026-11-22 one checkpoint said "1 entries added, 0 changed" while 3 904
epoch-less entries had left the file: the headline hid the largest change of the commit.

Injection: the dropped count removed from the headline -> this test fails.
"""
import json
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(TOOLS))
import certified_checkpoint as CP  # noqa: E402


def _sh(*cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True).stdout


def test_the_headline_says_how_many_entries_left_the_file(tmp_path):
    r = tmp_path / "repo"; r.mkdir()
    _sh("git", "init", "-q", "-b", "certify-branch", cwd=r)
    _sh("git", "config", "user.email", "t@t", cwd=r); _sh("git", "config", "user.name", "t", cwd=r)
    d = r / "src/neurobrix/config/autotune/v/p"; d.mkdir(parents=True)
    f = d / "k.fp32.json"
    f.write_text(json.dumps({"entries": {f"({i},)": {"config": i} for i in range(5)}}))
    _sh("git", "add", ".", cwd=r); _sh("git", "commit", "-q", "-m", "base", cwd=r)
    f.write_text(json.dumps({"entries": {"(9,)": {"config": 9}}}))          # +1, the 5 others dropped
    gate = tmp_path / "gate.py"; gate.write_text("import sys; sys.exit(0)\n")
    rec = tmp_path / "RUN.md"
    res = CP.checkpoint(str(r), "src/neurobrix/config/autotune", [], [sys.executable, str(gate)], [],
                        record=str(rec), say=lambda *a: None)
    assert res["committed"]
    head = _sh("git", "-C", str(r), "log", "-1", "--format=%s")
    assert "1 entries added, 0 changed, 5 dropped" in head, head
    assert "-5 dropped" in rec.read_text()
