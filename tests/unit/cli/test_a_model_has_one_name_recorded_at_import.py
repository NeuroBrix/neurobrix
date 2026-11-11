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


def test_an_entry_named_by_the_stamped_file_of_its_hub_record_is_found_installed(cache, monkeypatch, capsys):
    """The hub record's file carries a build stamp (`<name>.<YYYYMMDDTHHMMSS>.nbx`), the cache entry does not: the
    stem is compared without it, so the installed entry is found and nothing is downloaded (Wan2.2-I2V-A14B on the
    rack, 2026-11-08)."""
    import requests
    from neurobrix.cli.commands import registry
    monkeypatch.setattr(registry, "CACHE_DIR", cache)
    (cache / "Wan2.2-I2V-A14B").mkdir()
    (cache / "Wan2.2-I2V-A14B" / "manifest.json").write_text("{}")
    asked = []

    def get(url, **kw):
        asked.append(url)
        return _Resp({"model": {"slug": "Wan-AI/Wan2.2-I2V-A14B-Diffusers", "name": "Wan2.2-I2V-A14B-Diffusers",
                                "fileUrl": "models/Wan-AI/Wan2.2-I2V-A14B.20260912T223015.nbx"}})
    monkeypatch.setattr(requests, "get", get)
    with pytest.raises(SystemExit) as e:
        registry._import_body(_import_args("Wan-AI/Wan2.2-I2V-A14B-Diffusers"))
    assert e.value.code == 0 and len(asked) == 1, asked
    assert names.recorded_name(cache / "Wan2.2-I2V-A14B") == "Wan-AI/Wan2.2-I2V-A14B-Diffusers"


def test_the_command_line_refuses_the_entry_name_of_a_recorded_model_in_one_line(cache, tmp_path):
    """`run --model <entry>` for an entry whose hub name is recorded: exit 1, one line naming the hub name."""
    import os
    import subprocess
    import sys
    home = tmp_path / "home"
    shutil.copytree(cache, home / "cache")
    names.record_name(home / "cache" / "Kokoro-82M", "hexgrad/Kokoro-82M")
    repo = Path(__file__).resolve().parents[3]
    env = {**os.environ, "NEUROBRIX_HOME": str(home), "CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1",
           "PYTHONPATH": os.pathsep.join([str(repo / "src"), str(repo / "nbxtensor" / "src")])}
    p = subprocess.run([sys.executable, "-m", "neurobrix", "run", "--model", "Kokoro-82M", "--prompt", "hi"],
                       capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 1, p.stderr
    assert p.stderr.strip().splitlines() == [
        "Error: the model installed as Kokoro-82M is named hexgrad/Kokoro-82M: use --model hexgrad/Kokoro-82M"], p.stderr


def test_a_name_that_leads_to_a_recorded_entry_gets_the_one_message_and_not_found_lists_what_run_accepts(cache):
    """The new user's three tries (2026-11-11, on the published 0.6.5): the name typed at `import` answered "not
    found" with a list of FOLDER names, each of which `run` then refused. What this test does if the code is wrong:
    with the list read from the folders the second assertion fails; with only a bare directory name refused, the
    organisation-prefixed one returns "not found" and the first fails."""
    (cache / "Sana_1600M_1024px_MultiLing").mkdir()
    (cache / "Sana_1600M_1024px_MultiLing" / "manifest.json").write_text(json.dumps({"model_name": "x", "family": "image"}))
    names.record_name(cache / "Sana_1600M_1024px_MultiLing", "Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers")
    names.record_name(cache / "TinyLlama-1.1B-Chat-v1.0", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    for typed in ("Efficient-Large-Model/Sana_1600M_1024px_MultiLing", "Sana_1600M_1024px_MultiLing"):
        with pytest.raises(names.NameRefused, match="use --model Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers"):
            names.resolve(cache, typed)
    assert names.accepted_names(cache) == ["Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers", "Kokoro-82M",
                                           "TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
    for accepted in names.accepted_names(cache):                    # every listed name is one `--model` takes
        names.resolve(cache, accepted)
    assert names.resolve(cache, "Someone/Unknown") == ("Someone/Unknown", None)
    # another organisation's model of the same last name is ANOTHER model, not installed: "not found", never "is named"
    other = "somebody-else/TinyLlama-1.1B-Chat-v1.0"
    assert names.resolve(cache, other) == (other, None)
