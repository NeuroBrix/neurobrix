"""An entry installed under its container's own name (the manifest's model_name) is found by the container's sha256,
never by downloading the archive to read its manifest (reste_apres_064 point 13; agreed with the hub 2026-11-09 19:41).
Three rack entries (SANA-Video_2B_720p, Wan2.1-T2V-1.3B, Wan2.1-VACE-1.3B) are installed under names the hub record
never gives (slug, record name and file stem all end in _diffusers / -Diffusers): import would have downloaded the
archive to learn the name. Now: the name record carries the hub's container sha256; `--container FILE.nbx` verifies a
local copy against it and records the name with no download.

All network faked; a request for the download is an assertion error. What this test would do if the code were wrong:
a download for an entry already installed from the same container (red), a local file of other bytes accepted (red),
the user's own file deleted by --no-keep (red), the sha not recorded (red), an entry found by sha not recognised (red)."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest
import requests

import neurobrix.cli.commands.registry as reg
from neurobrix.nbx import names

ORG, NAME, ENTRY, REGISTRY = "acme", "Wan-Diffusers", "Wan", "https://registry.test"


def _archive(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"model_name": ENTRY, "name": ENTRY, "family": "video"}))
        zf.writestr("components/x/graph.json", "{}")
    return path


@pytest.fixture
def hub(tmp_path, monkeypatch):
    nbx = _archive(tmp_path / "copy_on_the_nas.nbx")
    sha = hashlib.sha256(nbx.read_bytes()).hexdigest()

    def fake_get(url, **kw):
        if url.endswith(f"/api/models/{ORG}/{NAME}"):
            return type("R", (), {"status_code": 200, "raise_for_status": lambda s: None,
                                  "json": lambda s: {"model": {"slug": f"{ORG}/{NAME}", "name": NAME, "sha256": sha,
                                                               "fileUrl": f"{REGISTRY}/f/{NAME}.20261009T000000.nbx",
                                                               "fileSize": nbx.stat().st_size}}})()
        raise AssertionError(f"downloaded: {url}")
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(reg, "_ACCEPTANCES_FILE", tmp_path / "acc.json")
    monkeypatch.setattr(reg, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(reg, "STORE_DIR", tmp_path / "store")
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    return nbx, sha, tmp_path / "cache"


def _installed(cache: Path, record=None) -> Path:
    e = cache / ENTRY
    (e / "components").mkdir(parents=True)
    (e / "manifest.json").write_text(json.dumps({"model_name": ENTRY, "family": "video"}))
    if record:
        (e / names.RECORD).write_text(json.dumps(record))
    return e


def _import(**kw):
    args = argparse.Namespace(model_ref=f"{ORG}/{NAME}", registry=REGISTRY, force=False, no_keep=False,
                              accept_license=True, json=False, container=None)
    for k, v in kw.items():
        setattr(args, k, v)
    try:
        reg.cmd_import(args)
    except SystemExit as e:
        return int(e.code or 0)
    return 0


def test_a_local_container_records_the_name_and_its_sha_with_no_download(hub):
    nbx, sha, cache = hub
    entry = _installed(cache)
    assert _import(container=str(nbx), no_keep=True) == 0
    rec = json.loads((entry / names.RECORD).read_text())
    assert rec["name"] == f"{ORG}/{NAME}" and rec["container_sha256"] == sha
    assert nbx.is_file(), "the user's own container was deleted"


def test_a_local_container_of_other_bytes_is_refused_naming_both(hub, caplog):
    nbx, sha, cache = hub
    entry = _installed(cache)
    other = nbx.parent / "other.nbx"
    other.write_bytes(nbx.read_bytes() + b"x")
    assert _import(container=str(other)) == 1
    said = caplog.text                                   # a refusal is a log record (registry._die)
    assert sha in said and hashlib.sha256(other.read_bytes()).hexdigest() in said
    assert not (entry / names.RECORD).exists()


def test_an_entry_recorded_with_the_hub_sha_is_found_by_it(hub, capsys):
    nbx, sha, cache = hub
    _installed(cache, record={"name": f"{ORG}/{NAME}", "container_sha256": sha})
    assert _import() == 0
    assert "already installed" in capsys.readouterr().out


def test_a_local_container_not_installed_is_installed_from_the_file(hub):
    nbx, sha, cache = hub
    assert _import(container=str(nbx), no_keep=True) == 0
    rec = json.loads((cache / ENTRY / names.RECORD).read_text())
    assert rec == {**rec, "name": f"{ORG}/{NAME}", "container_sha256": sha} and nbx.is_file()


def test_entry_of_container(tmp_path):
    e = _installed(tmp_path, record={"name": "a/b", "container_sha256": "f" * 64})
    assert names.entry_of_container(tmp_path, "f" * 64) == e and names.entry_of_container(tmp_path, "0" * 64) is None
