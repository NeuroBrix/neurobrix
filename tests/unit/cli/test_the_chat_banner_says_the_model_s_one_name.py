"""The chat's first line names the served model by its one name — the hub name `import` recorded beside its cache entry
(neurobrix/nbx/names.py) — never the entry's directory, which the daemon's status carries.

Injection (RED, then GREEN): the banner printing the status's entry name as given — the
recorded-name test fails.
"""
from __future__ import annotations

import json

from neurobrix.cli.commands import chat
from neurobrix.nbx import names


def _entry(cache, name):
    (cache / name).mkdir()
    (cache / name / "manifest.json").write_text(json.dumps({"model_name": name, "family": "llm"}))
    return cache / name


def test_the_banner_says_the_recorded_name_not_the_entry_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROBRIX_CACHE", str(tmp_path))
    names.record_name(_entry(tmp_path, "some-model-v1"), "some-org/some-model-v1")
    assert chat.banner("some-model-v1") == "some-org/some-model-v1 — /new /context /status /quit"


def test_an_entry_installed_before_the_record_keeps_its_directory_name(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROBRIX_CACHE", str(tmp_path))
    _entry(tmp_path, "older-model")
    assert chat.banner("older-model") == "older-model — /new /context /status /quit"
