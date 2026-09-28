"""The host ledger admits cells in arrival order: no later cell passes a living earlier waiter.

Wan2.2-I2V-A14B-Diffusers asks 200 of the matrix's 201 GiB and so starts only on an empty ledger; with
two gates running, smaller cells kept slipping in and it waited 1 h 47 min on card 0 without starting
(2026-09-27). The waiting queue is its own file (`host_waiting.json`), so a runner on the older code keeps
reading a plain ledger. Before this branch nothing records a waiter: the first test fails (admitted).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import regression_matrix as R  # noqa: E402

GB = 1 << 30


def _roomy(monkeypatch):
    monkeypatch.setattr(R, "_host_bytes", lambda: 1000 * GB)
    monkeypatch.setattr(R, "_mem_available", lambda: 1000 * GB)


def test_a_later_cell_does_not_pass_a_living_earlier_waiter(tmp_path, monkeypatch):
    _roomy(monkeypatch)
    waiter = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (tmp_path / "host_waiting.json").write_text(json.dumps({str(waiter.pid): time.time() - 60}))
        assert R.reserve_host(tmp_path, 1 * GB) is False
        assert str(os.getpid()) in json.loads((tmp_path / "host_waiting.json").read_text())
    finally:
        waiter.kill(); waiter.wait()


def test_a_dead_waiter_blocks_nobody_and_the_admitted_leave_the_queue(tmp_path, monkeypatch):
    _roomy(monkeypatch)
    dead = subprocess.Popen([sys.executable, "-c", "pass"]); dead.wait()
    (tmp_path / "host_waiting.json").write_text(json.dumps({str(dead.pid): time.time() - 60}))
    assert R.reserve_host(tmp_path, 1 * GB) is True
    assert json.loads((tmp_path / "host_waiting.json").read_text()) == {}


def test_the_first_waiter_is_admitted_once_there_is_room(tmp_path, monkeypatch):
    _roomy(monkeypatch)
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:   # a living cell holds 799 of the 800 GB budget: 2 GB cannot pass yet, and waits
        (tmp_path / "host_ledger.json").write_text(json.dumps({str(holder.pid): 799 * GB}))
        assert R.reserve_host(tmp_path, 2 * GB) is False
        since = json.loads((tmp_path / "host_waiting.json").read_text())[str(os.getpid())]
    finally:
        holder.kill(); holder.wait()
    assert R.reserve_host(tmp_path, 2 * GB) is True      # its turn kept, then admitted
    assert since < time.time() and json.loads((tmp_path / "host_waiting.json").read_text()) == {}
