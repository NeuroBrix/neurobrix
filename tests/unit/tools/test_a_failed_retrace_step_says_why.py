"""A retrace step that fails records WHY, read from its own log (2026-09-29: E2's trace failed in 4 s
on a missing `libunified_pool.so` and the retrace summary said only "failed"; the supervisor found
the cause by hand). What the test would do if the tool swallowed the cause again: the cause read from
the log would not be the ERROR line (seen red with `failure_cause` returning the last line only).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import retrace_zoo as RZ  # noqa: E402

E2_LOG = """$ /home/mlops/ml/venv/bin/python forge.py trace --model Wan2.1-VACE-1.3B-diffusers --family video --device cuda:0
[Disk] root filesystem 78.6 G free, floor 60 G — the trace may start

ERROR: libunified_pool.so not found at forge/vmm/libunified_pool.so. Run 'make' in forge/vmm to build it.
"""


def test_the_cause_is_the_error_line(tmp_path):
    p = tmp_path / "trace.log"
    p.write_text(E2_LOG + "\n[teardown] pool released\n")
    assert RZ.failure_cause(p).startswith("ERROR: libunified_pool.so not found")


def test_a_log_without_an_error_line_gives_its_last_lines(tmp_path):
    p = tmp_path / "build.log"
    p.write_text("step one\nstep two\nstep three\n")
    assert RZ.failure_cause(p) == "step one | step two | step three"


def test_an_unreadable_log_is_said(tmp_path):
    assert "could not be read" in RZ.failure_cause(tmp_path / "absent.log")
