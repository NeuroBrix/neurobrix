"""A model's `.running` lock names its holder's pid; the campaign trusts the lock only while
that pid exists. After a power loss (2026-09-07 12:16: Flex.1-alpha on GPU3, CogVideoX-2b on
GPU1) the dead runs' locks stayed on disk and, read as "running elsewhere", would have skipped
both models on every resume."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import precision_zoo_campaign as C  # noqa: E402


def test_lock_of_a_live_pid_is_held():
    assert C.lock_holder_alive(f"gpu=1 pid={os.getpid()} 11:32:43")


def test_lock_of_a_dead_pid_is_stale():
    dead = 2 ** 22 - 7                      # above the default pid_max, never a live process
    assert not Path(f"/proc/{dead}").exists()
    assert not C.lock_holder_alive(f"gpu=1 pid={dead} 11:32:43")


def test_lock_without_a_pid_is_trusted():
    assert C.lock_holder_alive("gpu=1 11:32:43")
