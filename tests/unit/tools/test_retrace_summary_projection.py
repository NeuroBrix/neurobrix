"""summary.json is a projection of every model's state.json on disk, rebuilt at every write:
an instance never overwrites a line another instance advanced (2026-09-07 13:08, the upload
loop's ok lost under a policy pass's stale DEFERRED)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import retrace_zoo as R  # noqa: E402


def _state(out: Path, model: str, steps: dict):
    (out / model).mkdir(parents=True, exist_ok=True)
    (out / model / "state.json").write_text(json.dumps({"model": model, "steps": steps}))


def test_summary_projects_every_state_file_and_replaces_stale_lines(tmp_path):
    out = tmp_path
    (out / "summary.json").write_text(json.dumps({"Kokoro-82M": {"gate": "PASS", "upload": "DEFERRED"},
                                                  "ghost": {"gate": "PASS"}}))
    _state(out, "Kokoro-82M", {"gate": {"ok": True, "verdict": "PASS"}, "upload": {"ok": True}})
    _state(out, "Voxtral", {"trace": {"ok": True}, "build": {"ok": False, "state": "DEFERRED", "reason": "disk"}})
    s = R.write_summary(out)
    on_disk = json.loads((out / "summary.json").read_text())
    assert s == on_disk
    assert on_disk["Kokoro-82M"] == {"gate": "PASS", "upload": "ok"}, "the state on disk wins over the stale line"
    assert on_disk["Voxtral"] == {"trace": "ok", "build": "DEFERRED"}
    assert "ghost" not in on_disk, "a line without a state file is not a model of the campaign"


def test_summary_line_prefers_verdict_then_state_then_ok(tmp_path):
    assert R.summary_line({"gate": {"ok": False, "verdict": "FAIL", "state": "x"}}) == {"gate": "FAIL"}
    assert R.summary_line({"upload": {"ok": False, "state": "REFUSED"}}) == {"upload": "REFUSED"}
    assert R.summary_line({"trace": {"ok": False}}) == {"trace": "failed"}
