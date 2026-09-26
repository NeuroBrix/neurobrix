"""A cell SIGKILLed before its timeout is written as a KILL, never as a TIMEOUT.

`run_group` returns -9 for a timeout, and a process killed by anything else (the kernel's OOM
killer) returns -9 too; the log said "TIMEOUT after 3600s" for a cell killed at 323 s
(2026-09-26). On the old `run` the first test fails (the log says TIMEOUT).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import precision_zoo_campaign as C  # noqa: E402


def test_a_sigkill_before_the_timeout_is_a_kill(tmp_path):
    log = tmp_path / "cell.log"
    rc, wall = C.run([sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"],
                     None, log, timeout=60)
    text = log.read_text()
    assert rc == -9 and wall < 60
    assert "KILLED by SIGKILL" in text and "TIMEOUT" not in text


def test_a_timeout_is_a_timeout(tmp_path):
    log = tmp_path / "cell.log"
    rc, _ = C.run([sys.executable, "-c", "import time; time.sleep(30)"], None, log, timeout=1)
    assert rc == -9 and "TIMEOUT after 1s" in log.read_text()
