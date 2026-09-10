"""The flight recorder writes the flight to the path it was given.

On 2026-09-10 a campaign ran for six hours under `flightrec run … --log
validation_outputs/…/rest.log`. That file received its last line on 9 September
at 04:25, while the per-model `result.json` files filled normally throughout.
Six hours of narration — which cell started when, which arm failed and why,
which settings were served — went to the terminal of an ssh session that then
died, and nothing on disk carried it.

The cause is not a crash. `--log` was documented as "path to the job's own log
file, for the resume block": flightrec RECORDED the path and never wrote to it,
launching the child as `subprocess.Popen(args.cmd)` with no capture, so the
child simply inherited whatever stdout the wrapper had.

A flight recorder that does not record the flight is the same family as a
metric reading an absent key: it produces silence, and silence is
indistinguishable from a quiet flight.

Run: PYTHONPATH=src python -m pytest tests/unit/tools/test_flightrec_records_the_flight.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[3] / "tools" / "flightrec.py"


def _run(tmp_path: Path, log: Path, *cmd):
    return subprocess.run(
        [sys.executable, str(TOOL), "run", "--label", "test-recording",
         "--log", str(log), "--"] + list(cmd),
        capture_output=True, text=True, cwd=TOOL.parent.parent,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
             "NBX_FLIGHTREC_DIR": str(tmp_path / "rec")})


def test_the_child_output_reaches_the_log(tmp_path):
    log = tmp_path / "flight.log"
    out = _run(tmp_path, log, "/bin/echo", "a-line-from-the-flight")
    assert log.exists(), (
        f"--log names a path the recorder never wrote (exit {out.returncode}): "
        f"{out.stdout[-400:]} {out.stderr[-400:]}")
    assert "a-line-from-the-flight" in log.read_text(), (
        f"the flight's own output is absent from {log}: {log.read_text()[:400]}")


def test_the_child_output_still_reaches_the_terminal(tmp_path):
    """Teeing, not swallowing: a live operator must still see the flight."""
    log = tmp_path / "flight.log"
    out = _run(tmp_path, log, "/bin/echo", "visible-on-the-terminal")
    assert "visible-on-the-terminal" in out.stdout, (
        f"the recorder swallowed the child's output instead of teeing it: "
        f"{out.stdout[-400:]}")


def test_the_exit_code_is_still_the_child_s(tmp_path):
    """Capturing must not change what the wrapper reports."""
    log = tmp_path / "flight.log"
    out = _run(tmp_path, log, "/bin/sh", "-c", "exit 3")
    assert out.returncode == 3, f"expected the child's 3, got {out.returncode}"
