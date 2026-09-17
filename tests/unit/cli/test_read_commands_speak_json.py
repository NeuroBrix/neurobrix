"""Every read command speaks one JSON record on stdout under `--json` — the
contract a client (the TUI, Studio) reads with zero engine logic. Each record
carries `schema` ("neurobrix.<command>/<version>", the version table in
`neurobrix/cli/json_out.py`) and `engine`; human lines go to stderr. The
test refuses an output that does not parse, or one that parses but carries no
schema — a record that reads as prose is a client that guesses.

Injection (2026-09-16): a stray `print("hello")` on stdout inside `info`'s
JSON branch made `test_info_speaks_one_record` RED (the output no longer
parsed); removed, green.
"""
from __future__ import annotations

import io
import json
import os
import sys
import contextlib
from pathlib import Path

import pytest

from neurobrix.cli import main
from neurobrix.cli.json_out import SCHEMAS

CACHE = Path(os.path.expanduser("~/.neurobrix/cache"))


def _run(argv):
    """The CLI in-process; (stdout, stderr, rc)."""
    out, err = io.StringIO(), io.StringIO()
    rc = 0
    old_argv, old_out = sys.argv, sys.stdout
    sys.argv = ["neurobrix"] + argv
    try:
        with contextlib.redirect_stderr(err):
            sys.stdout = out
            try:
                r = main()
                rc = 0 if r is None else int(r)
            except SystemExit as e:
                rc = int(e.code or 0)
    finally:
        sys.argv, sys.stdout = old_argv, old_out
    return out.getvalue(), err.getvalue(), rc


def _record(argv, command):
    out, err, rc = _run(argv)
    try:
        rec = json.loads(out)
    except json.JSONDecodeError as e:
        pytest.fail(f"{' '.join(argv)}: stdout is not one JSON record ({e}); stdout was:\n{out[:800]}")
    assert rec.get("schema") == f"neurobrix.{command}/{SCHEMAS[command]}", rec.get("schema")
    assert "engine" in rec
    return rec, err, rc


def test_info_speaks_one_record():
    rec, _, _ = _record(["info", "--json"], "info")
    for k in ("version", "cache", "store", "models", "hardware_profiles", "gpus"):
        assert k in rec, k
    assert isinstance(rec["models"], list)


def test_list_speaks_one_record():
    rec, _, _ = _record(["list", "--json"], "list")
    assert isinstance(rec["models"], list) and "store" in rec
    if rec["models"]:
        assert {"name", "family", "size_bytes", "license", "in_store"} <= set(rec["models"][0])


def test_autotune_status_and_check_speak_records():
    rec, _, _ = _record(["autotune", "status", "--json"], "autotune.status")
    assert "shapes" in rec and "served_by_memory_class_gb" in rec
    rec, _, rc = _record(["autotune", "check", "--json"], "autotune.check")
    assert isinstance(rec["files"], list) and "refused" in rec


def test_doctor_speaks_one_record_and_keeps_its_diagnosis_on_stderr():
    rec, err, _ = _record(["doctor", "--json"], "doctor")
    assert "ok" in rec and isinstance(rec["problems"], list)
    assert "Compute environment" in err, "the human diagnosis must go to stderr, not vanish"


@pytest.mark.skipif(not CACHE.exists() or not any(CACHE.glob("*/manifest.json")), reason="no installed container")
def test_inspect_and_coverage_speak_records():
    model = sorted(p.parent.name for p in CACHE.glob("*/manifest.json"))[0]
    rec, _, _ = _record(["inspect", model, "--json"], "inspect")
    assert rec["path"] and isinstance(rec["components"], list)
    rec, _, _ = _record(["coverage", "--json"], "coverage")
    assert rec["mode"] == "index" and rec["distinct_ops"] > 0
    rec, _, _ = _record(["coverage", "aten::mm", "--json"], "coverage")
    assert rec["mode"] == "symbol" and isinstance(rec["carried_by"], list)


def test_hub_speaks_one_record_from_the_registry_answer(monkeypatch):
    import urllib.request
    payload = json.dumps({"models": [{"slug": "acme/tiny", "name": "tiny", "category": "llm", "fileSize": 12,
                                      "license": "apache-2.0", "downloadCount": 3, "visibility": "PUBLIC"}],
                          "total": 1}).encode()

    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def getcode(self): return 200
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp(payload))
    rec, err, _ = _record(["hub", "--json"], "hub")
    assert rec["total"] == 1 and rec["models"][0]["slug"] == "acme/tiny"
    assert rec["models"][0]["installed"] is False


def test_every_schema_is_versioned_and_named():
    for cmd, v in SCHEMAS.items():
        assert isinstance(v, int) and v >= 1 and "/" not in cmd


@pytest.mark.skipif(not (CACHE / "TinyLlama-1.1B-Chat-v1.0" / "manifest.json").exists(), reason="TinyLlama not installed")
def test_explain_plan_speaks_one_record():
    rec, _, _ = _record(["run", "--model", "TinyLlama-1.1B-Chat-v1.0", "--prompt", "Hello", "--max-tokens", "4",
                         "--explain-plan", "--json"], "explain-plan")
    assert rec["strategy"] and isinstance(rec["components"], list) and rec["components"]
    assert {"name", "devices", "dtype"} <= set(rec["components"][0])
