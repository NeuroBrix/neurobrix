"""A model's one name is the hub name `import` records beside its cache entry (neurobrix/nbx/names.py), read by
`--model`, `list`, `remove` and the API's model id. An entry installed before 0.6.3 (no record) keeps its directory
name with a migration note — never two names at once.

Injections (each RED, then GREEN; see the commit message):
  * the record not written when import finds the entry already installed — the already-installed test fails;
  * the directory name accepted beside a recorded hub name (no NameRefused) — the one-name test fails;
  * served_name returning the entry name — the API-id test fails.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from neurobrix.nbx import names


@pytest.fixture
def cache():
    d = Path(tempfile.mkdtemp(prefix="nbxn"))
    for entry in ("TinyLlama-1.1B-Chat-v1.0", "Kokoro-82M"):
        (d / entry).mkdir()
        (d / entry / "manifest.json").write_text(json.dumps({"model_name": entry, "family": "llm"}))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_a_recorded_entry_has_exactly_one_name(cache):
    names.record_name(cache / "TinyLlama-1.1B-Chat-v1.0", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    assert names.resolve(cache, "TinyLlama/TinyLlama-1.1B-Chat-v1.0") == ("TinyLlama-1.1B-Chat-v1.0", None)
    with pytest.raises(names.NameRefused, match="named TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        names.resolve(cache, "TinyLlama-1.1B-Chat-v1.0")
    assert names.name_of(cache, "TinyLlama-1.1B-Chat-v1.0") == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def test_an_entry_installed_before_the_record_keeps_its_name_with_the_migration_note(cache):
    entry, note = names.resolve(cache, "Kokoro-82M")
    assert entry == "Kokoro-82M"
    assert note == "Kokoro-82M was installed before 0.6.3: run `neurobrix import <org>/Kokoro-82M` once to record its name"
    assert names.name_of(cache, "Kokoro-82M") == "Kokoro-82M"
    assert names.resolve(cache, "nobody/none") == ("nobody/none", None)


def test_a_malformed_record_is_refused_by_name_never_read_as_absent(cache):
    (cache / "Kokoro-82M" / names.RECORD).write_text("{}")
    with pytest.raises(RuntimeError, match="unreadable"):
        names.recorded_name(cache / "Kokoro-82M")


def _import_args(ref):
    return SimpleNamespace(model_ref=ref, registry="http://registry.invalid", force=False, no_keep=False,
                           json=False, yes=True)


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


def test_import_records_the_name_when_the_entry_is_already_installed(cache, monkeypatch, capsys):
    import requests
    from neurobrix.cli.commands import registry
    monkeypatch.setattr(registry, "CACHE_DIR", cache)
    monkeypatch.setattr(requests, "get", lambda url, **kw: _Resp(
        {"model": {"slug": "hexgrad/Kokoro-82M", "name": "Kokoro-82M", "fileUrl": "x/Kokoro-82M.nbx"}}))
    with pytest.raises(SystemExit) as e:
        registry._import_body(_import_args("hexgrad/Kokoro-82M"))
    assert e.value.code == 0
    assert names.recorded_name(cache / "Kokoro-82M") == "hexgrad/Kokoro-82M"
    assert "hexgrad/Kokoro-82M" in capsys.readouterr().out
    # a second import answers from the record, with no network
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    with pytest.raises(SystemExit):
        registry._import_body(_import_args("hexgrad/Kokoro-82M"))


def test_list_prints_the_one_name_and_the_note_with_the_hub_name(cache, monkeypatch):
    import requests
    from neurobrix.cli.commands import registry
    monkeypatch.setattr(registry, "CACHE_DIR", cache)
    monkeypatch.setattr(registry, "STORE_DIR", cache / "store")
    names.record_name(cache / "TinyLlama-1.1B-Chat-v1.0", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    monkeypatch.setattr(requests, "get", lambda url, **kw: _Resp(
        {"models": [{"slug": "hexgrad/Kokoro-82M", "name": "Kokoro-82M", "fileUrl": "Kokoro-82M.20260907T064009.nbx"}]}))
    rec = registry.list_record(SimpleNamespace(store=False, json=True))
    by_entry = {m["entry"]: m for m in rec["models"]}
    assert by_entry["TinyLlama-1.1B-Chat-v1.0"]["name"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert by_entry["TinyLlama-1.1B-Chat-v1.0"]["note"] is None
    assert by_entry["Kokoro-82M"]["name"] == "Kokoro-82M"
    assert "neurobrix import hexgrad/Kokoro-82M" in by_entry["Kokoro-82M"]["note"]


def test_the_api_model_id_is_the_recorded_name(cache, monkeypatch):
    from neurobrix.core import paths
    from neurobrix.serving.server import ServingDaemon
    monkeypatch.setattr(paths, "cache_dir", lambda: cache)
    names.record_name(cache / "TinyLlama-1.1B-Chat-v1.0", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    engine = SimpleNamespace(model_name="TinyLlama-1.1B-Chat-v1.0")
    assert ServingDaemon("x", "none", engine=engine).served_name() == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    engine.model_name = "Kokoro-82M"
    assert ServingDaemon("x", "none", engine=engine).served_name() == "Kokoro-82M"
