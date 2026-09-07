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
                                 restore_mbps=10.0, upload_mbps=10.0)


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
    monkeypatch.setattr(requests, "get", lambda url, stream=False, timeout=None, headers=None: _Resp())
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
    monkeypatch.setattr(requests, "get", lambda url, stream=False, timeout=None, headers=None: _Resp())
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
    monkeypatch.setattr(R, "export_readers", lambda cmdlines=None: [])
    monkeypatch.setattr(R, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("nothing must be streamed")))
    assert m.step_upload() is False
    st = m.state["steps"]["upload"]
    assert st["state"] == "DEFERRED" and "SlowDownWrite" in st["reason"]


def test_an_upload_is_paced_through_the_toolchains_flag(model, monkeypatch):
    m = model
    m.hub = "o/n"
    m.state["steps"]["build"] = {"ok": True, "nbx": "/x/model.nbx"}
    monkeypatch.setattr(R.repo_env, "require", lambda name: None)
    monkeypatch.setenv("NEUROBRIX_API_TOKEN", "t")
    monkeypatch.setattr(R, "hub_store_health", lambda: 200)
    monkeypatch.setattr(R, "hub_store_write_probe", lambda org, name, token: 200)
    monkeypatch.setattr(R, "export_readers", lambda cmdlines=None: [])
    seen = {}
    monkeypatch.setattr(R, "run", lambda cmd, *a, **k: seen.setdefault("cmd", [str(c) for c in cmd]) and 1)
    m.step_upload()
    assert seen["cmd"][-2:] == ["--max-write-mbps", "10.0"] and "replace" in seen["cmd"]


def _armed_for_upload(m, monkeypatch):
    m.state["steps"]["build"] = {"ok": True, "nbx": "/x/model.nbx"}
    monkeypatch.setattr(R.repo_env, "require", lambda name: None)
    monkeypatch.setenv("NEUROBRIX_API_TOKEN", "t")
    monkeypatch.setattr(R, "hub_store_health", lambda: 200)
    monkeypatch.setattr(R, "hub_store_write_probe", lambda org, name, token: 200)
    monkeypatch.setattr(R, "export_readers", lambda cmdlines=None: [])


def test_a_new_publication_carries_its_written_entry(model, monkeypatch, tmp_path):
    m = model; m.hub = None
    _armed_for_upload(m, monkeypatch)
    entries = tmp_path / "new_entries.json"
    entries.write_text(json.dumps({m.name: {"org": "o", "name": "N", "category": "TTS", "license": "apache-2.0",
                                            "tags": "a,b", "description": "d"}}))
    monkeypatch.setattr(R, "NEW_ENTRIES", entries)
    monkeypatch.setattr(R, "HUB_MAP", tmp_path / "hub_map.json")
    seen = {}
    monkeypatch.setattr(R, "run", lambda cmd, *a, **k: seen.setdefault("cmd", [str(c) for c in cmd]) and 0)
    monkeypatch.setattr(R.shutil, "rmtree", lambda *a, **k: None)
    m.step_upload()
    c = seen["cmd"]
    assert "publish" in c and c[c.index("--org") + 1] == "o" and c[c.index("--category") + 1] == "TTS"
    assert json.loads((tmp_path / "hub_map.json").read_text())[m.name] == "o/N"      # a later pass replaces


def test_a_container_with_neither_entry_nor_written_line_is_refused_by_name(model, monkeypatch, tmp_path):
    m = model; m.hub = None
    _armed_for_upload(m, monkeypatch)
    monkeypatch.setattr(R, "NEW_ENTRIES", tmp_path / "none.json")
    monkeypatch.setattr(R, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("nothing must run")))
    assert m.step_upload() is False
    st = m.state["steps"]["upload"]
    assert st["state"] == "REFUSED" and "no written new-entry line" in st["reason"]


def test_the_heavy_readers_of_the_export_are_named():
    cmds = ["/venv/bin/python /repo/forge/forge.py trace --model CogVideoX-2b --family video --device cuda:0 --path /snap/CogVideoX-2b",
            "python forge.py build --snapshot-path /hf/Janus-Pro-7B --family vlm --overwrite",
            "python tools/retrace_zoo.py --models x --gpu 0", "bash upload_loop.sh"]
    assert R.export_readers(cmds) == ["trace of CogVideoX-2b", "build of Janus-Pro-7B"]
    assert R.export_readers([]) == []


def test_an_upload_waits_for_a_window_with_no_reader(model, monkeypatch):
    m = model; m.hub = "o/n"
    _armed_for_upload(m, monkeypatch)
    monkeypatch.setattr(R, "export_readers", lambda cmdlines=None: ["trace of CogVideoX-2b"])
    monkeypatch.setattr(R, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("nothing must stream")))
    assert m.step_upload() is False
    st = m.state["steps"]["upload"]
    assert st["state"] == "DEFERRED" and "trace of CogVideoX-2b" in st["reason"]


def test_a_partial_previous_object_resumes_with_a_range(tmp_path, monkeypatch):
    import requests
    seen = {}

    class _Part(_Resp):
        status_code = 206
        def __init__(self, chunks): super().__init__(chunks)
    def get(url, stream=False, timeout=None, headers=None):
        seen["headers"] = headers or {}
        return _Part(10)
    monkeypatch.setattr(requests, "get", get)
    export = tmp_path / "export-a"; export.mkdir()
    monkeypatch.setattr(R, "SHARED_STORAGE_EXPORTS", (str(export),))
    import snapshot_refresh
    monkeypatch.setattr(snapshot_refresh, "_export_answers", lambda d, limit: 0.01)
    dest = tmp_path / "m.nbx"; dest.write_bytes(b"a" * 5000)                    # a stopped stream's partial file
    got = R.stream_under_probe("http://x", dest, 1000.0, tmp_path / "restore.log", probe_every=0.0, expected=5000 + 10 * 1024)
    assert seen["headers"] == {"Range": "bytes=5000-"} and got == 5000 + 10 * 1024
    assert dest.read_bytes()[:5000] == b"a" * 5000 and dest.stat().st_size == got
    assert R.stream_under_probe("http://x", dest, 1000.0, tmp_path / "restore.log", expected=got) == got   # complete: no request


def test_both_arms_run_under_one_frozen_autotune_state(model, monkeypatch, tmp_path):
    """One directory snapshot and one replay cache for the two arms; the stamps agree; a
    different stamp on one arm is refused by the gate by name."""
    m = model
    src = tmp_path / "tree" / "src"; d = src / "neurobrix" / "config" / "autotune" / "nvidia" / "volta"; d.mkdir(parents=True)
    (d / "matmul_kernel.fp32.json").write_text(json.dumps({"entries": {"k1": {}, "k2": {}}}))
    m.args.src = str(src)
    _manifest(R.CACHE / m.name, "T1"); _manifest(Path(m.args.backup) / m.name, "T1")
    seen = []
    def fake_run(cmd, env, logfile, timeout, cwd=None):
        seen.append((env.get("NEUROBRIX_AUTOTUNE_CERTIFIED_DIR"), env.get("NEUROBRIX_REPLAY_CACHE")))
        Path(str(logfile)).write_text("ok"); Path(str(cmd[cmd.index("--output") + 1])).write_bytes(b"out")
        return 0
    monkeypatch.setattr(R, "run", fake_run)
    monkeypatch.setattr(R, "sha", lambda p: "s")
    monkeypatch.setattr(C, "request_args", lambda name, fam, extra: [])
    monkeypatch.setattr(C, "output_ext", lambda fam, req: ".wav")
    assert m.step_old_outputs() is True
    fr = m.state["autotune_freeze"]
    assert fr["entries"] == 2 and Path(fr["directory"]).is_dir() and (Path(fr["directory"]) / "nvidia" / "volta" / "matmul_kernel.fp32.json").exists()
    assert seen and all(x == (fr["directory"], fr["replay"]) for x in seen)                    # both arms, same env
    assert m.state["steps"]["old_outputs"]["autotune"] == fr["snapshot"] and m.done("old_outputs")
    m.state["steps"]["new_outputs"] = {"ok": True, "policy": R.POLICY, "autotune": "another", "runs": {}}
    assert m.done("new_outputs") is False
    assert m.step_gate() is False and "different kernel-config states" in m.state["steps"]["gate"]["reason"]


def test_the_hubs_object_supersedes_a_backup_of_a_build_that_was_never_the_hubs(model, monkeypatch, tmp_path):
    import requests, zipfile
    m = model; m.hub = "o/n"
    _manifest(R.CACHE / m.name, "LOCAL-06-02"); _manifest(Path(m.args.backup) / m.name, "LOCAL-06-02")
    (Path(m.args.backup) / m.name / "components" / "core").mkdir(parents=True)
    monkeypatch.setattr(R, "hub_store_health", lambda: 200)
    monkeypatch.setattr(R.repo_env, "require", lambda name: None)
    monkeypatch.setenv("NEUROBRIX_API_TOKEN", "t")

    class _Rec:
        def raise_for_status(self): pass
        def json(self): return {"model": {"fileUrl": "models/o/n.nbx", "fileSize": 5, "updatedAt": "x"}}
    class _Url(_Rec):
        def json(self): return {"url": "http://read"}
    monkeypatch.setattr(requests, "get", lambda url, **k: _Url() if "admin" in url else _Rec())

    def stream(url, dest, mbps, logfile, expected=0):
        with zipfile.ZipFile(dest, "w") as zf:
            zf.writestr("manifest.json", json.dumps({"created_at": "HUB-06-09"}))
        return 5
    monkeypatch.setattr(R, "stream_under_probe", stream)
    monkeypatch.setattr(R, "run", lambda cmd, env, logfile, timeout, cwd=None: (_manifest(R.CACHE / m.name, "HUB-06-09"), 0)[1])
    assert m.restore_previous() is True
    bdir = Path(m.args.backup) / m.name
    assert json.loads((bdir / "manifest.json").read_text())["created_at"] == "HUB-06-09"          # the backup is the hub's graphs now
    aside = list(Path(m.args.backup).glob(f"{m.name}.local-build-*"))
    assert len(aside) == 1 and (aside[0] / "components" / "core").is_dir()                          # the local build kept aside
    po = m.state["steps"]["previous_object"]
    assert po["ok"] and po["supersedes"]["local_build"] == "LOCAL-06-02" and po["supersedes"]["hub_object"] == "HUB-06-09"
    assert m.state["steps"]["backup"]["supersedes"]["hub_object"] == "HUB-06-09"


def test_a_verdict_on_other_arms_than_the_stamped_ones_is_re_gated(model):
    """Parakeet 08:32: both arms re-run under the frozen state, the 07:32 PASS still counted as
    done and the chain moved on without a verdict on the new arms."""
    m = model
    m.state["autotune_freeze"] = {"snapshot": "S2"}
    m.state["steps"]["gate"] = {"ok": True, "verdict": "PASS", "policy": R.POLICY, "autotune": {"snapshot": "S1"}}
    assert m.done("gate") is False
    m.state["steps"]["gate"]["autotune"] = {"snapshot": "S2"}
    assert m.done("gate") is True
    m.state["steps"]["gate"]["policy"] = None
    assert m.done("gate") is False


def test_the_upload_loop_trusts_a_recorded_pass_whatever_the_gates_freshness(model, monkeypatch, tmp_path):
    """`--only-upload` uploads an artifact whose recorded verdict is PASS even when the chain would
    re-gate it (a verdict older than the frozen protocol): nine gated artifacts were refused at 08:34."""
    m = model
    m.state["steps"]["gate"] = {"ok": True, "verdict": "PASS"}           # no autotune stamp: the chain would re-gate
    assert m.done("gate") is False
    gate = m.state["steps"]["gate"]
    assert gate.get("verdict", "").startswith("PASS") and gate.get("ok") is True   # the loop's own test, as in main()


def test_a_hub_object_that_does_not_run_is_recorded_and_the_new_container_is_judged_on_its_own_engines(model, monkeypatch):
    m = model
    _manifest(R.CACHE / m.name, "T2"); _manifest(Path(m.args.backup) / m.name, "T1")
    m.state["steps"]["build"] = {"ok": True, "nbx": "/x/model.nbx"}; m.state["steps"]["install"] = {"ok": True}
    monkeypatch.setattr(m, "restore_previous", lambda: True)
    monkeypatch.setattr(m, "reinstall_new", lambda: True)
    (m.dir / "old_sequential.log").write_text("x\n[ERROR] Pipeline failed: ZERO FALLBACK: No allocation for component 'perception_encoder'.\n")
    # a second attempt after the restore: the cache already holds the hub's object (previous_object ok), no restore this time
    m.state["steps"]["previous_object"] = {"ok": True}
    _manifest(R.CACHE / m.name, "T1")
    monkeypatch.setattr(m, "outputs", lambda tag: {"sequential": {"rc": 1, "sha": None, "output": str(m.dir / "o.txt")},
                                                    "triton": {"rc": 1, "sha": None, "output": str(m.dir / "t.txt")}})
    assert m.step_old_outputs() is True                                    # the fact is recorded, the chain goes on
    so = m.state["steps"]["old_outputs"]
    assert so["ok"] and "perception_encoder" in so["unrunnable"]
    fr = m.state["autotune_freeze"]["snapshot"]
    m.state["steps"]["new_outputs"] = {"ok": True, "policy": R.POLICY, "autotune": fr,
                                       "runs": {"sequential": {"rc": 0, "sha": "abc", "output": "s"}, "triton": {"rc": 0, "sha": "abc", "output": "t"}}}
    monkeypatch.setattr(m, "graph_diff", lambda: {"components": {}, "beyond_annotation": 0, "annotation_changes": 3, "arg_witnessed": 1,
                                                  "pruned_dead_ops": 0, "corrupted_before": 2, "corrupted_after": 0})
    assert m.step_gate() is True
    g = m.state["steps"]["gate"]
    assert g["verdict"].startswith("PASS (the hub's object does not run") and g["bytes"]["new engines"] == "IDENTICAL"
    m.state["steps"].pop("gate")
    m.state["steps"]["new_outputs"]["runs"]["triton"]["sha"] = "zzz"
    monkeypatch.setattr(C, "gate", lambda a, b: {"kind": "text", "identical": False})
    assert m.step_gate() is False and m.state["steps"]["gate"]["verdict"] == "NEEDS_EXPLANATION"
