"""The certifier's door must see a checkpointer on a platform with no /proc.

`cli/commands/autotune.py::_checkpointer_holds` walks `/proc` to find a running
`certified_checkpoint.py` holding this repository. It reads /proc rather than calling `pgrep`
on purpose — so the check cannot match its OWN command line, a self-match that has cost this
session three shells.

macOS has no /proc: `os.listdir("/proc")` raises `FileNotFoundError`, which surfaced as the
certifier dying with `ERROR: [Errno 2] No such file or directory: '/proc'` the moment a
checkpointer was finally holding the repository (2026-09-22). The door could not see, so
nothing could be certified here.

The self-match protection must survive the port: the scan skips this process's own pid.
"""
from __future__ import annotations

import subprocess
import sys
import time

import pytest

from neurobrix.cli.commands import autotune as A


@pytest.fixture
def fake_checkpointer(tmp_path):
    """A process whose argv[1] really is a path ending in certified_checkpoint.py."""
    script = tmp_path / "certified_checkpoint.py"
    script.write_text("import sys, time\ntime.sleep(60)\n")
    repo = tmp_path / "a_repo"
    repo.mkdir()
    p = subprocess.Popen([sys.executable, str(script), "--repo", str(repo)])
    time.sleep(0.6)
    try:
        yield repo
    finally:
        p.kill()
        p.wait()


def test_the_door_sees_a_checkpointer_holding_this_repo(fake_checkpointer):
    assert A._checkpointer_holds(fake_checkpointer) is True


def test_the_door_does_not_see_one_holding_another_repo(fake_checkpointer, tmp_path):
    other = tmp_path / "another_repo"
    other.mkdir()
    assert A._checkpointer_holds(other) is False


def test_the_door_answers_false_with_no_checkpointer_at_all(tmp_path):
    lonely = tmp_path / "unheld"
    lonely.mkdir()
    assert A._checkpointer_holds(lonely) is False
