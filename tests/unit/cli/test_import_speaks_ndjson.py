"""`neurobrix import --json` is a lifecycle a client reads: one NDJSON event
per phase on stdout — info, (license), download with the real byte counts,
downloaded, extracting, installed, done — or a terminal `error`; every human
line on stderr; the model visible under its name only after the extraction
succeeded (it extracts into `<name>.installing` and renames in one motion).
`remove --json` answers with one record of what it removed.
All network is faked (`requests.get`), no GPU (Studio requests 5 and 6).

Injections: with the staging rename replaced by a direct extraction the
visibility test failed (a manifest under the final name before the end);
with `_die` printing instead of emitting, the refusal test read no `error`.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest
import requests

import neurobrix.cli.commands.registry as reg

ORG, NAME, REGISTRY = "acme", "tiny-1b", "https://registry.test"


class _Resp:
    def __init__(self, payload=None, status=200, body=b"", headers=None):
        self._payload, self.status_code, self._body = payload, status, body
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)

    def json(self):
        return self._payload

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]


def _archive(tmp_path: Path, manifest=None) -> bytes:
    p = tmp_path / "src.nbx"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest or {"name": NAME, "family": "llm"}))
        zf.writestr("components/x/graph.json", "{}")
    return p.read_bytes()


def _wire(monkeypatch, tmp_path, body: bytes, gated=False, status_info=200):
    def fake_get(url, **kw):
        if url.endswith(f"/api/models/{ORG}/{NAME}"):
            return _Resp({"model": {"fileSize": len(body), "category": "LLM", "license": "mit",
                                    "licenseName": "MIT", "gated": gated}}, status=status_info)
        if url.endswith("/download"):
            return _Resp({"url": f"{REGISTRY}/blob", "fileName": f"{NAME}.nbx"})
        if url.endswith("/blob"):
            return _Resp(body=body, headers={"content-length": str(len(body))})
        raise AssertionError(url)
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(reg, "_ACCEPTANCES_FILE", tmp_path / "acc.json")
    monkeypatch.setattr(reg, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(reg, "STORE_DIR", tmp_path / "store")
    monkeypatch.setattr(sys, "stdin", io.StringIO())


def _args(**kw):
    base = dict(model_ref=f"{ORG}/{NAME}", registry=REGISTRY, force=False, no_keep=False,
                accept_license=False, json=True)
    base.update(kw)
    return argparse.Namespace(**base)


def _run(capsys, args):
    code = 0
    try:
        reg.cmd_import(args)
    except SystemExit as e:
        code = int(e.code or 0)
    out, err = capsys.readouterr()
    events = [json.loads(line) for line in out.splitlines() if line.strip()]
    return code, events, err


def test_the_lifecycle_is_one_event_per_phase_and_nothing_else_on_stdout(tmp_path, monkeypatch, capsys):
    _wire(monkeypatch, tmp_path, _archive(tmp_path))
    code, events, err = _run(capsys, _args())
    assert code == 0
    names = [e["event"] for e in events]
    assert names[0] == "info" and names[-1] == "done"
    assert [n for n in names if n != "download"] == ["info", "downloaded", "extracting", "installed", "done"]
    assert all(e["schema"] == "neurobrix.import/1" for e in events)
    dl = [e for e in events if e["event"] == "download"]
    assert dl[-1]["bytes"] == dl[-1]["total"] == len(_archive(tmp_path))
    assert "IMPORT COMPLETE" in err          # the human lines went to stderr
    assert (tmp_path / "cache" / NAME / "manifest.json").exists()
    assert not reg.installing_path(tmp_path / "cache" / NAME).exists()


def test_the_model_is_visible_under_its_name_only_after_extraction_succeeded(tmp_path, monkeypatch, capsys):
    _wire(monkeypatch, tmp_path, _archive(tmp_path))
    final = tmp_path / "cache" / NAME
    seen = {}
    real = reg.extract_container

    def spy(store_path, cache_path):
        seen["target"] = Path(cache_path)
        seen["final_visible_during"] = final.exists()
        real(store_path, cache_path)
        seen["final_visible_after_extract"] = final.exists()
    monkeypatch.setattr(reg, "extract_container", spy)
    code, events, _ = _run(capsys, _args())
    assert code == 0
    assert seen["target"] == reg.installing_path(final)
    assert seen["final_visible_during"] is False and seen["final_visible_after_extract"] is False
    assert (final / "manifest.json").exists()


def test_a_failed_extraction_leaves_no_half_model_behind(tmp_path, monkeypatch, capsys):
    _wire(monkeypatch, tmp_path, _archive(tmp_path))
    final = tmp_path / "cache" / NAME

    def boom(store_path, cache_path):
        # `exist_ok=True`, like the real `extract_container` it stands in for:
        # the staging directory is created by the install brick before the
        # extractor is called, not by the extractor itself.
        Path(cache_path).mkdir(parents=True, exist_ok=True)
        (Path(cache_path) / "manifest.json").write_text("{}")
        raise ValueError("member refused")
    monkeypatch.setattr(reg, "extract_container", boom)
    with pytest.raises(ValueError):
        reg.cmd_import(_args())
    assert not final.exists() and not reg.installing_path(final).exists()


def test_a_refusal_is_one_error_event_and_exit_1(tmp_path, monkeypatch, capsys):
    _wire(monkeypatch, tmp_path, _archive(tmp_path), status_info=404)
    code, events, err = _run(capsys, _args())
    assert code == 1
    assert [e["event"] for e in events] == ["error"]
    assert "not found on registry" in events[0]["message"]
    assert "ERROR" in err


def test_a_gated_model_under_json_is_refused_by_name_never_prompted(tmp_path, monkeypatch, capsys):
    _wire(monkeypatch, tmp_path, _archive(tmp_path), gated=True)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("a client was prompted"))
    code, events, _ = _run(capsys, _args())
    assert code == 1 and events[-1]["event"] == "error" and "--accept-license" in events[-1]["message"]
    code, events, _ = _run(capsys, _args(accept_license=True))
    assert code == 0
    assert [e for e in events if e["event"] == "license"][0]["accepted_via"] == "--accept-license"


def test_a_model_being_installed_is_not_listed(tmp_path, monkeypatch):
    monkeypatch.setattr(reg, "CACHE_DIR", tmp_path / "cache")
    for d in ("ready", "half.installing"):
        (tmp_path / "cache" / d).mkdir(parents=True)
        (tmp_path / "cache" / d / "manifest.json").write_text(json.dumps({"name": d, "family": "llm"}))
    names = [m["name"] for m in reg.list_record(argparse.Namespace())["models"]]
    assert "ready" in names and not any(n.endswith(".installing") for n in names)


def test_remove_answers_with_one_record(tmp_path, monkeypatch, capsys):
    _wire(monkeypatch, tmp_path, _archive(tmp_path))
    _run(capsys, _args())
    try:
        reg.cmd_remove(argparse.Namespace(model_name=NAME, store=False, all=True, json=True))
    except SystemExit:
        pass
    out, err = capsys.readouterr()
    rec = json.loads(out)
    assert rec["schema"] == "neurobrix.remove/1" and rec["found"] is True
    assert sorted(r["kind"] for r in rec["removed"]) == ["cache", "store"]
    assert all(r["bytes"] > 0 for r in rec["removed"])
    assert "Removed cache" in err


def test_the_progress_bar_speaks_ndjson_at_most_once_per_interval_and_at_close(capsys):
    from neurobrix.cli.json_out import NdjsonProgress
    t = [0.0]
    bar = NdjsonProgress(total=100, initial=10, desc="x.nbx", every=1.0, clock=lambda: t[0])
    bar.update(5); t[0] = 0.5; bar.update(5)          # inside the interval: silent
    t[0] = 1.5; bar.update(10)                         # past it: one event
    bar.close()                                        # always at close
    events = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert [e["bytes"] for e in events] == [10, 30, 30]
    assert events[0]["total"] == 100 and events[0]["file"] == "x.nbx"
