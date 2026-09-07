"""The retrace gate's two arms run under ONE precision policy, and the old arm runs on the OLD container.

Outputs carry the policy they were measured under; ones measured under another policy are set
aside and re-run, never compared. When the cache holds another build than the backed-up one (a
state reset for a re-trace drops its install step while the cache keeps the build), the old arm
runs on the hub's previous object, brought back through the standard install and put back
afterwards. A read of that object stops by name when a shared export stalls.
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


def _args(tmp):
    return types.SimpleNamespace(out=str(tmp / "out"), backup=str(tmp / "backup"), models_root=str(tmp / "builds"),
                                 tmp=str(tmp / "tmp"), gpu=None, src=None, extra=[], timeout=10, trace_timeout=10,
                                 restore_mbps=10.0)


def _manifest(path: Path, created: str):
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(json.dumps({"created_at": created}))


def _fake_outputs(m, calls):
    def outputs(tag):
        calls.append(tag)
        for arm in ("sequential", "triton"):
            (m.dir / f"{tag}_{arm}.wav").write_bytes(b"x" + tag.encode())
            (m.dir / f"{tag}_{arm}.log").write_text("ok")
        return {"sequential": {"rc": 0, "sha": "a"}, "triton": {"rc": 0, "sha": "b"}}
    return outputs


@pytest.fixture
def model(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "family_of", lambda n: "tts")
    monkeypatch.setattr(R, "CACHE", tmp_path / "cache")
    return R.Model("kokoro-test", _args(tmp_path))


def test_old_outputs_measured_under_another_policy_are_set_aside_and_re_run(model, monkeypatch):
    m, calls = model, []
    _manifest(R.CACHE / m.name, "T1"); _manifest(Path(m.args.backup) / m.name, "T1")   # the cache holds the backed-up container
    (m.dir / "old_sequential.wav").write_bytes(b"old"); (m.dir / "old_sequential.log").write_text("x")
    m.state["steps"]["old_outputs"] = {"ok": True, "runs": {}}                          # no stamp: the default policy of before
    assert m.done("old_outputs") is False
    monkeypatch.setattr(m, "outputs", _fake_outputs(m, calls))
    assert m.step_old_outputs() is True
    assert calls == ["old"]
    aside = list(m.dir.glob("superseded_*_old"))
    assert len(aside) == 1 and (aside[0] / "old_sequential.wav").read_bytes() == b"old"
    assert "default" in (aside[0] / "WHY.txt").read_text() and R.POLICY in (aside[0] / "WHY.txt").read_text()
    st = m.state["steps"]["old_outputs"]
    assert st["policy"] == R.POLICY and st["container"] == "the installed container" and m.done("old_outputs") is True
    assert m.step_old_outputs() is True and calls == ["old"]                             # stamped: not run again


def test_the_old_arm_runs_on_the_previous_object_when_the_cache_holds_another_build(model, monkeypatch):
    m, calls = model, []
    _manifest(R.CACHE / m.name, "T2"); _manifest(Path(m.args.backup) / m.name, "T1")   # the retraced build sits in the cache
    m.state["steps"]["build"] = {"ok": True, "nbx": "/x/model.nbx"}
    m.state["steps"]["install"] = {"ok": True}
    monkeypatch.setattr(m, "restore_previous", lambda: calls.append("restore") or True)
    monkeypatch.setattr(m, "reinstall_new", lambda: calls.append("reinstall") or True)
    monkeypatch.setattr(m, "outputs", _fake_outputs(m, calls))
    assert m.step_old_outputs() is True
    assert calls == ["restore", "old", "reinstall"]
    assert m.state["steps"]["old_outputs"]["container"] == "the hub's previous object"


def test_a_reset_state_still_sees_the_other_build_in_the_cache(model, monkeypatch):
    """No install step in the state (reset for a re-trace), yet the cache holds the pass's build:
    the previous object is restored and, with nothing to put back, stays until the chain installs."""
    m, calls = model, []
    _manifest(R.CACHE / m.name, "T2"); _manifest(Path(m.args.backup) / m.name, "T1")
    monkeypatch.setattr(m, "restore_previous", lambda: calls.append("restore") or True)
    monkeypatch.setattr(m, "reinstall_new", lambda: calls.append("reinstall") or True)
    monkeypatch.setattr(m, "outputs", _fake_outputs(m, calls))
    assert m.step_old_outputs() is True and calls == ["restore", "old"]


def test_without_a_backup_the_installed_container_is_the_old_arm(model, monkeypatch):
    m, calls = model, []
    _manifest(R.CACHE / m.name, "T1")                                                   # first pass: no backup yet
    monkeypatch.setattr(m, "restore_previous", lambda: calls.append("restore") or True)
    monkeypatch.setattr(m, "outputs", _fake_outputs(m, calls))
    assert m.step_old_outputs() is True and calls == ["old"]


def test_new_outputs_measured_under_another_policy_are_set_aside_too(model, monkeypatch):
    m, calls = model, []
    (m.dir / "new_triton.wav").write_bytes(b"new")
    m.state["steps"]["new_outputs"] = {"ok": True, "runs": {}}
    assert m.done("new_outputs") is False
    monkeypatch.setattr(m, "outputs", _fake_outputs(m, calls))
    assert m.step_new_outputs() is True and calls == ["new"]
    aside = list(m.dir.glob("superseded_*_new"))
    assert len(aside) == 1 and "default" in (aside[0] / "WHY.txt").read_text()
    assert m.state["steps"]["new_outputs"]["policy"] == R.POLICY


def test_the_gate_refuses_two_arms_under_different_policies(model):
    m = model
    m.state["steps"]["old_outputs"] = {"ok": True, "policy": R.POLICY, "runs": {}}
    m.state["steps"]["new_outputs"] = {"ok": True, "runs": {}}
    assert m.step_gate() is False
    g = m.state["steps"]["gate"]
    assert g["verdict"] == "FAIL" and "different precision policies" in g["reason"] and "'default'" in g["reason"]


class _Resp:
    """What `requests.get(url, stream=True)` gives: 49 chunks of a kilobyte."""
    def __init__(self, chunks=49):
        self.chunks = chunks

    def raise_for_status(self): pass

    def iter_content(self, k):
        for _ in range(self.chunks):
            yield b"z" * 1024

    def __enter__(self): return self
    def __exit__(self, *a): return False


def test_a_read_of_the_previous_object_stops_by_name_when_an_export_stalls(tmp_path, monkeypatch):
    import requests
    import snapshot_refresh
    monkeypatch.setattr(requests, "get", lambda url, stream=False, timeout=None: _Resp())
    export = tmp_path / "export-a"; export.mkdir()
    monkeypatch.setattr(R, "SHARED_STORAGE_EXPORTS", (str(export),))
    log = tmp_path / "restore.log"
    monkeypatch.setattr(snapshot_refresh, "_export_answers", lambda d, limit: None)
    assert R.stream_under_probe("http://x", tmp_path / "m.nbx", 1000.0, log, probe_every=0.0) is None
    assert "export-a did not list" in log.read_text() and "http://x" not in log.read_text()
    monkeypatch.setattr(snapshot_refresh, "_export_answers", lambda d, limit: 0.01)
    assert R.stream_under_probe("http://x", tmp_path / "m.nbx", 1000.0, log, probe_every=0.0) == 49 * 1024
    assert (tmp_path / "m.nbx").stat().st_size == 49 * 1024


def test_a_hub_that_refuses_the_read_url_defers_by_name(model, monkeypatch):
    import requests
    m = model
    m.hub = "org/x"
    monkeypatch.setattr(R, "hub_store_health", lambda: 200)
    monkeypatch.setattr(R.repo_env, "require", lambda name: None)
    monkeypatch.setenv("NEUROBRIX_API_TOKEN", "t")

    class _Forbidden:
        status_code = 403
        def raise_for_status(self): raise requests.HTTPError("403 Client Error: Forbidden")
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Forbidden())
    assert m.restore_previous() is False
    st = m.state["steps"]["old_outputs"]
    assert st["state"] == "DEFERRED" and "read URL" in st["reason"] and "403" in st["reason"]


def test_every_shared_export_the_probe_lists_is_a_directory_here_or_the_probe_says_so(tmp_path, monkeypatch):
    """A path the probe cannot see must never read as a quiet export."""
    monkeypatch.setattr(R, "SHARED_STORAGE_EXPORTS", (str(tmp_path / "not-mounted"),))
    import urllib.request
    class _OK:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=None: _OK())
    assert "not mounted" in str(R.hub_store_health())
    import requests
    monkeypatch.setattr(requests, "get", lambda url, stream=False, timeout=None: _Resp())
    log = tmp_path / "restore.log"
    assert R.stream_under_probe("http://x", tmp_path / "m.nbx", 1000.0, log, probe_every=0.0) is None
    assert "not-mounted did not list" in log.read_text()


def test_the_write_probe_names_the_stores_refusal_before_any_artifact_streams(monkeypatch):
    import requests
    calls = []

    class _Slot:
        def raise_for_status(self): pass
        def json(self): return {"key": "models/o/n.probe.nbx", "uploadUrl": "http://store/put"}

    class _Put:
        status_code = 503
        text = "<Error><Code>SlowDownWrite</Code><Message>Resource requested is unwritable, please reduce your request rate</Message></Error>"
    monkeypatch.setattr(requests, "post", lambda *a, **k: calls.append("slot") or _Slot())
    monkeypatch.setattr(requests, "put", lambda *a, **k: calls.append("put") or _Put())
    monkeypatch.setattr(requests, "delete", lambda *a, **k: calls.append(("drop", k.get("params"))) or None)
    answer = R.hub_store_write_probe("o", "n", "t")
    assert answer == "503 SlowDownWrite: Resource requested is unwritable, please reduce your request rate"
    assert calls == ["slot", "put", ("drop", {"key": "models/o/n.probe.nbx"})]
    _Put.status_code = 200
    assert R.hub_store_write_probe("o", "n", "t") == 200


def test_an_upload_is_deferred_by_the_stores_name_when_the_write_probe_fails(model, monkeypatch):
    m = model
    m.hub = "o/n"
    m.state["steps"]["build"] = {"ok": True, "nbx": "/x/model.nbx"}
    monkeypatch.setattr(R.repo_env, "require", lambda name: None)
    monkeypatch.setenv("NEUROBRIX_API_TOKEN", "t")
    monkeypatch.setattr(R, "hub_store_health", lambda: 200)
    monkeypatch.setattr(R, "hub_store_write_probe", lambda org, name, token: "503 SlowDownWrite")
    monkeypatch.setattr(R, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("nothing must be streamed")))
    assert m.step_upload() is False
    st = m.state["steps"]["upload"]
    assert st["state"] == "DEFERRED" and "SlowDownWrite" in st["reason"]
