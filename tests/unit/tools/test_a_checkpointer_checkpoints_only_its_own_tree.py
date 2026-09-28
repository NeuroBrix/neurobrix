"""A checkpointer checkpoints only the tree it lives in.

2026-09-28 18:55: two certification chains launched as tmux windows (cwd: the main checkout) ran
`tools/certified_checkpoint.py` by a relative path after a backgrounded `cd` — main's copy, older
than the certify tree's, without the 30-minute push ration — and two unattended pushes of one
branch went out 97 s apart. The rules live in the file; a foreign copy applies foreign rules.

The door: a repo that carries its OWN, different `tools/certified_checkpoint.py` is checkpointed only
by that copy; a scratch repo with none (every other test here) passes.

What this test would do if the code were wrong: without the door the foreign run proceeds (rc 0,
`--once` on a clean tree) and the first assertion fails (seen red with the door's call removed).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[3] / "tools" / "certified_checkpoint.py"


def _sh(*cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True).stdout


def test_a_repo_other_than_the_tools_own_tree_is_refused(tmp_path):
    other = tmp_path / "other"; other.mkdir()
    _sh("git", "init", "-q", "-b", "certify-branch", cwd=other)
    (other / "tools").mkdir()
    (other / "tools" / "certified_checkpoint.py").write_text("# another tree's copy, with rules of its own\n")
    r = subprocess.run([sys.executable, str(TOOL), "--repo", str(other), "--once", "--remotes", ""],
                       capture_output=True, text=True)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "REFUSED" in r.stderr and str(other) in r.stderr


def test_a_scratch_repo_without_a_copy_passes_the_door(tmp_path):
    sys.path.insert(0, str(TOOL.parent))
    import certified_checkpoint as C
    _sh("git", "init", "-q", cwd=tmp_path)
    assert C.refuse_a_repo_this_tool_is_not_from(str(tmp_path)) == ""


def test_the_tools_own_tree_passes_the_door():
    sys.path.insert(0, str(TOOL.parent))
    import certified_checkpoint as C
    assert C.refuse_a_repo_this_tool_is_not_from(str(TOOL.parents[1])) == ""
