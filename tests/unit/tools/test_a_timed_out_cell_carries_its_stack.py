"""A TIMEOUT row carries the stack the cell was in when its budget ran out.

The post-fix matrix of 2026-09-27 came back with seven TIMEOUT rows whose only evidence
was the last line of output ("[Triton] 180 constant(s) bound in fp32 once", "'lm_head':
loading 1 weights") — a line printed minutes before the kill, which names no cause. A
stack read by hand needs the cell alive, so every TIMEOUT cost a live re-run. The stack
is now taken by the tool itself, in the window where the group is still alive, and the
row names the frames.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(TOOLS))

import precision_zoo_campaign as Z  # noqa: E402
import regression_matrix as R  # noqa: E402


def test_before_kill_runs_while_the_group_is_alive(tmp_path):
    seen = {}

    def before_kill(pgid):
        seen["alive"] = [p for p in Z.group_pids(pgid) if Path(f"/proc/{p}").exists()]

    with open(tmp_path / "log", "w") as fh:
        rc = Z.run_group([sys.executable, "-c", "import time; time.sleep(60)"], dict(os.environ), fh, 2,
                         before_kill=before_kill)
    assert rc == -9
    assert seen.get("alive"), "the hook ran after the group was gone — nothing left to read"


def test_the_stack_section_is_never_empty(tmp_path, monkeypatch):
    """No py-spy reachable: the section says so rather than staying silent."""
    monkeypatch.setenv("PATH", "/nonexistent")
    import sysconfig
    monkeypatch.setattr(sysconfig, "get_path", lambda *a, **k: "/nonexistent-too")
    with open(tmp_path / "log", "w") as fh:
        Z.dump_group_stacks(os.getpgid(0), fh)
    text = (tmp_path / "log").read_text()
    assert Z.STACK_MARK in text and "no stack: py-spy is not on" in text


def test_a_parsed_stack_names_the_engine_frames_innermost_first(tmp_path):
    log = tmp_path / "cell.log"
    log.write_text(
        "[Triton] 180 constant(s) bound in fp32 once\n"
        f"\n{Z.STACK_MARK}\n--- pid 7 (python) rc=0\n"
        "Process 7: python -m neurobrix run\nPython v3.10.12\n\n"
        'Thread 7 (active): "MainThread"\n'
        "    _launch (neurobrix/kernels/ops/matmul.py:812)\n"
        "    run (neurobrix/triton/sequence.py:2201)\n"
        "    _run_code (runpy.py:86)\n"
        '\nThread 9 (idle): "pin"\n    wait (threading.py:320)\n'
        "\nTIMEOUT after 900s\n")
    assert R.stack_at_timeout(log) == ["_launch (neurobrix/kernels/ops/matmul.py:812)",
                                       "run (neurobrix/triton/sequence.py:2201)"]
    assert R.last_stage(log)["last_line"] == "[Triton] 180 constant(s) bound in fp32 once", (
        "the stack section is not the cell's output — the last line is read above it")


@pytest.mark.skipif(subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode != 0,
                    reason="needs passwordless sudo for py-spy")
def test_a_real_timeout_writes_a_real_stack(tmp_path):
    log = tmp_path / "cell.log"
    prog = "import time\ndef held_here():\n    time.sleep(60)\nheld_here()\n"
    rc, _ = Z.run([sys.executable, "-c", prog], dict(os.environ), log, 3, stack_at_timeout=True)
    assert rc == -9
    text = log.read_text()
    assert Z.STACK_MARK in text
    assert "held_here" in text, text[-800:]
