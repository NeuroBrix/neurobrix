"""A resumed chain measures the container it BUILT, never the one the cache happens to hold.

granite-speech-3.3-8b, 2026-09-26 02:43 UTC: the chain's state said trace, build and install were
done; the shared cache had meanwhile been put back to the hub's previous object by hand (the
build had shipped no `modules/` and every run failed). Relaunched, the chain skipped the install
because the state said so, ran `new_outputs` on the PREVIOUS container and handed the gate the
June graph on both sides — a verdict about nothing, read green. `cache_holds_backup()` already
answers the question the state cannot; the install step now asks it before trusting its mark,
reinstalls the recorded `.nbx`, and drops the arms and verdict measured on the wrong container.

Seen RED on retrace-2026-09-26 before the fix: the first cell found no install command and the
stale `new_outputs` and `gate` marks still in the state.
"""
from __future__ import annotations

import json
import sys
import types
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import retrace_zoo as R  # noqa: E402
import precision_zoo_campaign as C  # noqa: E402

NAME = "audio-test"


def _args(tmp):
    return types.SimpleNamespace(out=str(tmp / "out"), backup=str(tmp / "backup"), models_root=str(tmp / "builds"),
                                 tmp=str(tmp / "tmp"), gpu=None, src=None, extra=[], timeout=10, trace_timeout=10,
                                 restore_mbps=10.0, upload_mbps=10.0)


def _manifest(root: Path, created_at: str):
    (root / NAME).mkdir(parents=True, exist_ok=True)
    (root / NAME / "manifest.json").write_text(json.dumps({"model_name": NAME, "created_at": created_at}))


@pytest.fixture
def chain(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "family_of", lambda n: "audio_llm")
    monkeypatch.setattr(R, "CACHE", tmp_path / "cache")
    nbx = tmp_path / "builds" / "audio_llm" / NAME / "model.nbx"
    nbx.parent.mkdir(parents=True)
    with zipfile.ZipFile(nbx, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"model_name": NAME, "created_at": "2026-09-26T02:18:21"}))
    m = R.Model(NAME, _args(tmp_path))
    m.state["steps"] = {
        "trace": {"ok": True}, "build": {"ok": True, "nbx": str(nbx)},
        "install": {"ok": True, "installed_name": NAME},
        "new_outputs": {"ok": True, "runs": {}}, "gate": {"ok": True, "verdict": "PASS"},
    }
    installs = []

    def fake_run(cmd, env, log, timeout, cwd=None):
        installs.append([str(c) for c in cmd])
        _manifest(R.CACHE, "2026-09-26T02:18:21")          # what `forge local --overwrite` leaves behind
        Path(log).write_text("installed")
        return 0
    monkeypatch.setattr(R, "run", fake_run)
    monkeypatch.setattr(m, "env", lambda tree=False: {})
    return m, installs


def test_an_install_marked_done_over_the_previous_container_is_redone_and_the_stale_arms_dropped(chain, tmp_path):
    m, installs = chain
    _manifest(R.CACHE, "2026-06-04T18:46:52")               # the hub's previous object, put back by hand
    _manifest(Path(m.args.backup), "2026-06-04T18:46:52")   # the backup holds the same build
    assert m.cache_holds_backup() is True
    assert m.step_install() is True
    assert len(installs) == 1 and "local" in installs[0] and "--overwrite" in installs[0]
    assert m.cache_holds_backup() is False                  # the built container is what the arms will run on
    assert "new_outputs" not in m.state["steps"] and "gate" not in m.state["steps"]
    assert m.state["steps"]["install"]["ok"] is True and m.state["steps"]["install"].get("reinstalled")


def test_an_install_marked_done_over_the_built_container_is_trusted(chain):
    m, installs = chain
    _manifest(R.CACHE, "2026-09-26T02:18:21")               # the cache holds the build
    _manifest(Path(m.args.backup), "2026-06-04T18:46:52")
    assert m.step_install() is True
    assert installs == [] and "gate" in m.state["steps"]


def test_without_a_backup_the_mark_is_trusted(chain):
    m, installs = chain
    _manifest(R.CACHE, "2026-09-26T02:18:21")               # no backup: nothing to compare with
    assert m.cache_holds_backup() is None
    assert m.step_install() is True and installs == []
