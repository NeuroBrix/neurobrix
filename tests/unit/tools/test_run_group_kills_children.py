"""A campaign command runs in its own process group: a timeout kills its children too."""
from __future__ import annotations

import io
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import precision_zoo_campaign as C  # noqa: E402


def test_a_timeout_kills_the_command_and_its_child(tmp_path):
    pidfile = tmp_path / "child.pid"
    with open(tmp_path / "log", "w") as fh:
        rc = C.run_group(["bash", "-c", f"sleep 60 & echo $! > {pidfile}; wait"], dict(os.environ), fh, timeout=1)
    assert rc == -9
    child = int(pidfile.read_text())
    for _ in range(50):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        raise AssertionError("the child outlived its parent's timeout")


def test_a_command_that_ends_in_time_returns_its_code(tmp_path):
    with open(tmp_path / "log", "w") as fh:
        assert C.run_group(["bash", "-c", "exit 3"], dict(os.environ), fh, timeout=5) == 3
