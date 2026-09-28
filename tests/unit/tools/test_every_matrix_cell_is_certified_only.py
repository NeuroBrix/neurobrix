"""Every matrix cell is a confirmation run: its command carries `--certified-only`.

The owner's method (2026-09-28 20:06): matrices, gates and verifications CONFIRM, served entirely from
the certified directory; a missing key is an error naming its census row, never a runtime sweep. The
engine's mode is `neurobrix run --certified-only` (a-confirmation-run-is-certified-only); this tool
is the runner that must always pass it.

What this test would do if the code were wrong: without the flag in the cell's command the assertion
fails (seen red with the flag removed from `_run_cell`).
"""
from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(TOOLS))
import regression_matrix as R  # noqa: E402


def test_the_cell_command_is_certified_only(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, env, log, timeout, **kw):
        seen["cmd"] = list(cmd)
        Path(log).write_text("")                  # the cell's log, as the real runner leaves one
        return 1, 0.1

    monkeypatch.setattr(R.Z, "run", fake_run)
    monkeypatch.setattr(R.Z, "family_of", lambda model: "llm")
    monkeypatch.setattr(R.Z, "output_ext", lambda family, req: ".txt")
    monkeypatch.setattr(R, "cell_request", lambda model: ["--prompt", "x"])
    monkeypatch.setattr(R, "off_trace_size", lambda model, family: None)
    for mode in R.MODES:
        R._run_cell("M", mode, "0", tmp_path, 60, tmp_path / "src")
        assert "--certified-only" in seen["cmd"], (mode, seen["cmd"])
