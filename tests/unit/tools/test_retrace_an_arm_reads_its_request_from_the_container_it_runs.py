"""An arm reads its request from the container it RUNS, and a step that raises leaves a record.

Wan2.1-T2V-1.3B-Diffusers and Wan2.1-VACE-1.3B-diffusers, retrace6 chain3f, 2026-10-12 03:37 / 03:43:
the install installed the build under the registry entry (Wan2.1-T2V-1.3B) and RETIRED the cached
name; `new_outputs` then built its request from the retired name's topology (a video family asks it
whether the model takes an image) and raised. The process died rc=1, the state ended at `install`,
and summary.json kept the previous attempt's `trace: failed` — the failure was nowhere but a log.

Seen RED on eed0845f: the first test raised the ZERO FALLBACK of `_declares_image_input` on the
retired name, the second found the state without a `new_outputs` record and the summary stale.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import retrace_zoo as R  # noqa: E402
import precision_zoo_campaign as C  # noqa: E402

CACHED = "Vid-1B-Diffusers"     # the name the cache held before the install
ENTRY = "Vid-1B"                # the registry entry the install wrote, retiring CACHED


def _args(tmp):
    return types.SimpleNamespace(stage=str(tmp / "stage"), out=str(tmp / "out"), backup=str(tmp / "backup"),
                                 models_root=str(tmp / "builds"), tmp=str(tmp / "tmp"), gpu=None, src=None, extra=[],
                                 timeout=10, trace_timeout=10, restore_mbps=10.0, upload_mbps=10.0)


@pytest.fixture
def installed(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    (cache / ENTRY).mkdir(parents=True)               # only the entry: the cached name is retired
    (cache / ENTRY / "topology.json").write_text(json.dumps({"inputs": ["global.image"]}))
    monkeypatch.setattr(R, "CACHE", cache)
    monkeypatch.setattr(C, "CACHE", cache)
    monkeypatch.setattr(C, "family_of", lambda n: "video")
    monkeypatch.setattr(R, "registry_entry_name", lambda n: ENTRY)
    out = tmp_path / "out" / CACHED
    out.mkdir(parents=True)
    (out / "state.json").write_text(json.dumps({"model": CACHED, "steps": {
        "old_outputs": {"ok": True}, "trace": {"ok": True}, "build": {"ok": True}, "backup": {"ok": True},
        "install": {"ok": True, "installed_name": ENTRY}}}))
    runs = []

    def fake_run(cmd, env, log, timeout, cwd=None):
        cmd = [str(c) for c in cmd]
        runs.append(cmd)
        Path(cmd[cmd.index("--output") + 1]).write_bytes(b"video")
        Path(log).write_text("ok")
        return 0
    monkeypatch.setattr(R, "run", fake_run)
    return tmp_path, runs


def test_the_new_arm_reads_its_request_from_the_installed_entry(installed, monkeypatch):
    tmp, runs = installed
    m = R.Model(CACHED, _args(tmp))
    monkeypatch.setattr(m, "env", lambda tree=True: {})
    assert m.new_name == ENTRY
    res = m.outputs("new")
    assert {a: r["rc"] for a, r in res.items()} == {"sequential": 0, "triton": 0}
    for cmd in runs:
        assert cmd[cmd.index("--model") + 1] == ENTRY
        assert "--input-image" in cmd                     # the entry's topology was read: the image input kept
    assert "--input-image" in m.current_request()         # the request check is live, not "" (no check)


def test_a_step_that_raises_is_recorded_and_the_summary_rewritten(installed, monkeypatch):
    tmp, _ = installed
    (tmp / "out" / "summary.json").write_text(json.dumps({CACHED: {"old_outputs": "ok", "trace": "failed"}}))

    def boom(self):
        raise RuntimeError("the arm could not compose its request")
    monkeypatch.setattr(R.Model, "step_new_outputs", boom)
    monkeypatch.setattr(R.Model, "done", lambda self, step: step in ("old_outputs", "trace", "build", "backup", "install"))
    monkeypatch.setattr(sys, "argv", ["retrace_zoo.py", "--models", CACHED, "--out", str(tmp / "out"), "--stop-at", "gate"])
    with pytest.raises(RuntimeError, match="could not compose"):
        R.main()
    st = json.loads((tmp / "out" / CACHED / "state.json").read_text())["steps"]
    assert st["new_outputs"]["ok"] is False and "could not compose" in st["new_outputs"]["error"]
    assert "gate" not in st                               # the gate never ran, and the record says why
    summary = json.loads((tmp / "out" / "summary.json").read_text())
    assert summary[CACHED] == {"old_outputs": "ok", "trace": "ok", "build": "ok", "backup": "ok", "install": "ok",
                               "new_outputs": "failed"}
