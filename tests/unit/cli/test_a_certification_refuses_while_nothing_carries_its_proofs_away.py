"""A certification writes its proofs into the directory entry by entry, and nothing else
carries them anywhere. Three mains cuts (2026-09-11, 09-12, 09-13) each found hundreds of
certified entries on disk and nowhere else, which is why `tools/certified_checkpoint.py`
exists — and on 2026-09-22 eleven hours of stage two, 2 497 entries across seven files, stood
in the working tree alone with no checkpoint commit and no checkpoint ref on either remote.
The brick was there; nothing STARTED it. A brick that must be remembered is a brick that will
be forgotten, so the harmful STATE — certifying while no checkpointer holds the repository —
is made unreachable at entry instead of reported afterwards
(`docs/reference/proving-by-doors.md`).

What this test would do if the code were wrong: with the door removed the first case returns
"" and passes silently; with the repository path resolved one directory short — the bug this
door had for its first five minutes — the second case fails, because a checkpointer that IS
holding the repository would not be seen and every certification would refuse.

The processes here are real: a file named `certified_checkpoint.py` run with `--repo <dir>`,
so the check reads the same /proc it reads in production rather than a stand-in for it.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from neurobrix.cli.commands.autotune import (_checkpointer_holds, _refuse_without_a_checkpointer,
                                             _repo_root)


def test_the_repository_is_the_repository_not_its_source_directory():
    """Reads the path THE CODE uses, never one the test computes for itself — the first
    version of this cell recomputed `parents[4]` and stayed green under the very off-by-one
    it was written to catch."""
    repo = _repo_root()
    assert (repo / "tools" / "certified_checkpoint.py").exists(), repo
    assert (repo / "src" / "neurobrix").is_dir(), repo
    assert repo.name != "src", repo


def test_a_directory_no_process_holds_is_not_held(tmp_path):
    assert _checkpointer_holds(tmp_path) is False


def test_a_running_checkpointer_is_seen(tmp_path):
    fake = tmp_path / "certified_checkpoint.py"
    fake.write_text("import time, sys\ntime.sleep(30)\n")
    held = tmp_path / "a_repo"
    held.mkdir()
    proc = subprocess.Popen([sys.executable, str(fake), "--repo", str(held)])
    try:
        for _ in range(50):
            if _checkpointer_holds(held):
                break
            time.sleep(0.1)
        assert _checkpointer_holds(held) is True
        assert _checkpointer_holds(tmp_path / "another_repo") is False   # it holds ONE repository
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_the_refusal_names_the_tool_and_its_deliberate_opening(monkeypatch):
    monkeypatch.setattr("neurobrix.cli.commands.autotune._checkpointer_holds", lambda _r: False)
    said = _refuse_without_a_checkpointer(False)
    assert said.startswith("REFUSED:")
    assert "certified_checkpoint.py" in said and "--allow-uncheckpointed" in said
    assert "2026-09-11" in said, "the refusal carries what it costs, not just that it refuses"


def test_the_opening_is_an_opening(monkeypatch):
    monkeypatch.setattr("neurobrix.cli.commands.autotune._checkpointer_holds", lambda _r: False)
    assert _refuse_without_a_checkpointer(True) == ""
