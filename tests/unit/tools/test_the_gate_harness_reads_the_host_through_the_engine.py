"""The gate harness reads the host through readers every host answers — one implementation for both machines.

The Mac, 2026-09-29 16:02: every cell of its first gate through `tools/regression_matrix.py` was
refused in seconds — the harness read /proc/meminfo (and /proc/<pid>/status, /proc/<pid>/task/...
/children, /proc/<pid>) itself, and macOS has no /proc. The available figure now comes from the
engine's own reader (`core.host_memory.memory_state`), the process tree from psutil, liveness from
`os.kill(pid, 0)`.

What would this file do if the code were wrong? A /proc read left in the tool -> the first test,
RED; the available figure not the engine's -> the second, RED; the tree not counting a child ->
the third, RED.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))
import regression_matrix as R  # noqa: E402


def test_the_harness_names_no_linux_only_path():
    code = [l for l in (REPO / "tools" / "regression_matrix.py").read_text().splitlines()
            if "/proc" in l and not l.strip().startswith("#") and '"""' not in l]
    assert not [l for l in code if 'Path(f"/proc' in l or 'Path("/proc' in l or "open('/proc" in l], code


def test_the_available_figure_is_the_engine_s(monkeypatch):
    from neurobrix.core import host_memory as H
    monkeypatch.setattr(H, "memory_state", lambda: H.MemoryState(total_mb=1000, available_mb=321, source="stub"))
    assert R._mem_available() == 321 << 20
    monkeypatch.setattr(H, "memory_state", lambda: H.MemoryState(source="unreadable here"))
    try:
        R._mem_available()
    except SystemExit as e:
        assert "unreadable here" in str(e)
    else:
        raise AssertionError("an unreadable host was not refused")


def test_a_cell_s_resident_tree_counts_its_children():
    child = subprocess.Popen([sys.executable, "-c",
                              "b = bytearray(200 << 20)\nimport time\ntime.sleep(20)"])
    try:
        time.sleep(2.0)
        mine = R._rss_tree(os.getpid())
        assert mine - R._rss_tree(child.pid) > 0
        assert R._rss_tree(child.pid) >= 150 << 20
        assert R._children_rss() >= 150 << 20
    finally:
        child.kill()
