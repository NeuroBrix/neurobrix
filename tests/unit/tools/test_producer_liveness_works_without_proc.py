"""`producer_alive` must answer on a platform with no /proc.

`tools/wait_for.py` reads `/proc/<pid>/stat` for the producer's state and start time. macOS
and the BSDs have no `/proc`, so the read raises, `_stat_fields` returns None, and
`producer_alive()` answers **False for every pid, including a process that is plainly
running**.

The consequence is not cosmetic: `tools/certified_checkpoint.py` holds a producer and exits
when it is gone, so on this Mac it decided its producer was "already gone at start" every
time, ran one empty checkpoint and exited — and `cli/commands/autotune.py` then refuses to
certify at all, because `6442fe30` requires a checkpointer to be holding the repository.
Certification of the Apple directory was impossible until this was fixed (2026-09-22).

It failed CLOSED, which is the right direction, and that is exactly why it took a certification
launch to find it: nothing was ever wrong, only permanently refused.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import wait_for  # noqa: E402


def test_a_running_process_reads_as_alive():
    assert wait_for.producer_alive(os.getpid()) is True


def test_a_live_child_reads_as_alive():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        time.sleep(0.3)
        assert wait_for.producer_alive(p.pid) is True
    finally:
        p.kill()
        p.wait()


def test_a_dead_process_reads_as_gone():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    time.sleep(0.2)
    wait_for._BIRTH.pop(p.pid, None)
    assert wait_for.producer_alive(p.pid) is False


def test_a_pid_that_never_existed_reads_as_gone():
    assert wait_for.producer_alive(999_999) is False
