"""A chain launched through `tools/run_frozen.py` is not changed by an edit of its script.

2026-09-29: merge-queue-16's 32a gate ran `gate.sh` directly; the script was rewritten in place while
its loop ran, bash resumed at the old byte offset in the new text, the final `echo "gate 32a done"`
never ran, and the chain waiting for that marker refused. The test rebuilds that shape: a script
whose loop sleeps, then writes its marker; while the loop runs, a block is inserted INSIDE the
loop's text (the 16:28 edit's place), which moves every later byte.

What would this file do if the code were wrong? Launched on the source itself (the door removed)
the marker is never written -> the second test proves the shape is real (it asserts the failure),
and the first, RED.
"""
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools" / "run_frozen.py"

SCRIPT = """#!/bin/bash
M=$1
for i in 1 2 3; do
  sleep 0.5
done
echo "chain done" >> $M
"""
INSERT = "  # a later edit inside the loop, as at 16:28 — it moves every byte after it\n" * 6


def _edit_while_running(script: Path):
    time.sleep(0.3)                                  # the loop has been parsed and is sleeping
    text = script.read_text()
    i = text.index("  sleep 0.5")
    script.write_text(text[:i] + INSERT + text[i:])  # in place: same path, new bytes


def _run(cmd, script, marker):
    p = subprocess.Popen(cmd + [str(marker)])
    _edit_while_running(script)
    p.wait(20)
    return marker.read_text() if marker.exists() else ""


def test_a_frozen_chain_writes_its_marker_whatever_the_script_becomes(tmp_path):
    script = tmp_path / "chain.sh"
    script.write_text(SCRIPT)
    marker = tmp_path / "RUN.md"
    out = _run([sys.executable, str(TOOL), str(script)], script, marker)
    assert out == "chain done\n"
    frozen = list((tmp_path / ".frozen").glob("chain.*.sh"))
    assert len(frozen) == 1 and frozen[0].read_text() == SCRIPT      # the copy kept the bytes it ran
    assert not frozen[0].stat().st_mode & 0o222                        # read-only


def test_the_same_edit_on_a_script_run_directly_loses_the_marker(tmp_path):
    """The shape is real: the defect this door closes (no door, no marker)."""
    script = tmp_path / "chain.sh"
    script.write_text(SCRIPT)
    marker = tmp_path / "RUN.md"
    assert _run(["bash", str(script)], script, marker) == ""


def test_a_missing_script_is_refused_by_name(tmp_path):
    r = subprocess.run([sys.executable, str(TOOL), str(tmp_path / "absent.sh")], capture_output=True, text=True)
    assert r.returncode != 0 and "no such script" in r.stderr
