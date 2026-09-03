"""Non-interactive license acceptance for `neurobrix import` (gated models).

Pins the scripting/CI contract of the license gate:
  - non-interactive stdin without --accept-license / NBX_ACCEPT_LICENSE=1
    exits 1 with actionable guidance (license name, full URL, both
    non-interactive options) — never a mute "Declined";
  - --accept-license and NBX_ACCEPT_LICENSE=1 record the acceptance
    (keyed by org/name) and proceed past the gate without prompting;
  - legacy bare-name acceptance entries are honored AND migrated to the
    org/name key;
  - structured registry errors on the download-URL step are relayed
    (LICENSE_LOGIN_REQUIRED gets a dedicated explanation), with the raw
    print kept only for non-JSON bodies.

All network is mocked (requests.get monkeypatched). No GPU, no network.

Run: PYTHONPATH=src python -m pytest tests/unit/cli/test_import_license.py
"""
from __future__ import annotations

import argparse
import io
import json
import sys

import pytest
import requests

import neurobrix.cli.commands.registry as reg


ORG = "acme"
NAME = "gated-7b"
MODEL_REF = f"{ORG}/{NAME}"
REGISTRY = "https://registry.test"
LICENSE_ID = "test-license-1.0"
LICENSE_NAME = "Test License 1.0"
FULL_LICENSE_URL = f"{REGISTRY}/licenses/{LICENSE_ID}"

MODEL_INFO = {
    "model": {
        "fileSize": 1234,
        "category": "LLM",
        "description": "A gated test model",
        "license": LICENSE_ID,
        "licenseName": LICENSE_NAME,
        "licenseUrl": f"/licenses/{LICENSE_ID}",
        "gated": True,
    }
}


class _GatePassed(Exception):
    """Sentinel raised by the mocked download endpoint: the license gate
    was passed and cmd_import reached the download-URL step."""


class _FakeResponse:
    def __init__(self, payload=None, status_code=200, json_error=False):
        self._payload = payload
        self.status_code = status_code
        self._json_error = json_error

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)

    def json(self):
        if self._json_error:
            raise ValueError("response body is not JSON")
        return self._payload


class _NonTTYStdin(io.StringIO):
    def isatty(self):
        return False


class _TTYStdin(io.StringIO):
    def isatty(self):
        return True


def _install_registry_api(monkeypatch, download_behavior):
    """Mock requests.get: model metadata always answers; the download-URL
    endpoint delegates to download_behavior()."""

    def fake_get(url, *args, **kwargs):
        if url == f"{REGISTRY}/api/models/{ORG}/{NAME}":
            return _FakeResponse(MODEL_INFO)
        if url == f"{REGISTRY}/api/models/{ORG}/{NAME}/download":
            return download_behavior()
        raise AssertionError(f"unexpected URL fetched: {url}")

    monkeypatch.setattr(requests, "get", fake_get)


def _gate_passed():
    raise _GatePassed()


def _args(accept_license=False):
    return argparse.Namespace(
        model_ref=MODEL_REF,
        registry=REGISTRY,
        force=False,
        no_keep=False,
        accept_license=accept_license,
    )


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch, tmp_path):
    """Isolate acceptance file + cache/store dirs; clear the env override."""
    monkeypatch.setattr(reg, "_ACCEPTANCES_FILE", tmp_path / "license_acceptances.json")
    monkeypatch.setattr(reg, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(reg, "STORE_DIR", tmp_path / "store")
    monkeypatch.delenv("NBX_ACCEPT_LICENSE", raising=False)


# ── Non-interactive terminal without flag/env → guidance, exit 1 ──────


def test_noninteractive_without_flag_exits_with_guidance(monkeypatch, capsys):
    _install_registry_api(monkeypatch, lambda: pytest.fail("gate must block before download"))
    monkeypatch.setattr(sys, "stdin", _NonTTYStdin())

    with pytest.raises(SystemExit) as exc:
        reg.cmd_import(_args())
    assert exc.value.code == 1

    out = capsys.readouterr().out
    assert LICENSE_NAME in out
    assert FULL_LICENSE_URL in out
    assert "--accept-license" in out
    assert "NBX_ACCEPT_LICENSE=1" in out
    assert "Declined." not in out


def test_eof_on_prompt_prints_same_guidance(monkeypatch, capsys):
    """A TTY whose input() dies with EOFError gets the same guidance."""
    _install_registry_api(monkeypatch, lambda: pytest.fail("gate must block before download"))
    monkeypatch.setattr(sys, "stdin", _TTYStdin())

    def _eof(*_a, **_k):
        raise EOFError()

    monkeypatch.setattr("builtins.input", _eof)

    with pytest.raises(SystemExit) as exc:
        reg.cmd_import(_args())
    assert exc.value.code == 1

    out = capsys.readouterr().out
    assert "--accept-license" in out
    assert "NBX_ACCEPT_LICENSE=1" in out
    assert "Declined." not in out


# ── Flag / env acceptance → record + proceed, no prompt ───────────────


def test_accept_license_flag_records_and_proceeds(monkeypatch, capsys):
    _install_registry_api(monkeypatch, _gate_passed)
    monkeypatch.setattr(sys, "stdin", _NonTTYStdin())

    with pytest.raises(_GatePassed):
        reg.cmd_import(_args(accept_license=True))

    out = capsys.readouterr().out
    assert LICENSE_NAME in out
    assert FULL_LICENSE_URL in out
    assert "License accepted via --accept-license." in out

    data = json.loads(reg._ACCEPTANCES_FILE.read_text())
    assert data[MODEL_REF]["license"] == LICENSE_ID


def test_env_var_records_and_proceeds(monkeypatch, capsys):
    _install_registry_api(monkeypatch, _gate_passed)
    monkeypatch.setattr(sys, "stdin", _NonTTYStdin())
    monkeypatch.setenv("NBX_ACCEPT_LICENSE", "1")

    with pytest.raises(_GatePassed):
        reg.cmd_import(_args())

    out = capsys.readouterr().out
    assert "License accepted via NBX_ACCEPT_LICENSE=1." in out

    data = json.loads(reg._ACCEPTANCES_FILE.read_text())
    assert data[MODEL_REF]["license"] == LICENSE_ID


# ── Legacy bare-name cache entries: honored + migrated to org/name ────


def test_legacy_bare_name_entry_honored_and_migrated(monkeypatch, capsys):
    reg._ACCEPTANCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    reg._ACCEPTANCES_FILE.write_text(json.dumps(
        {NAME: {"license": LICENSE_ID, "accepted_at": "2026-01-01T00:00:00+00:00"}}
    ))
    _install_registry_api(monkeypatch, _gate_passed)
    monkeypatch.setattr(sys, "stdin", _NonTTYStdin())

    # No flag, no env, no TTY — the legacy acceptance must pass the gate.
    with pytest.raises(_GatePassed):
        reg.cmd_import(_args())

    data = json.loads(reg._ACCEPTANCES_FILE.read_text())
    assert MODEL_REF in data
    assert NAME not in data
    assert data[MODEL_REF]["license"] == LICENSE_ID


def test_is_license_accepted_migrates_bare_key_directly():
    reg._ACCEPTANCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    reg._ACCEPTANCES_FILE.write_text(json.dumps(
        {NAME: {"license": LICENSE_ID, "accepted_at": "2026-01-01T00:00:00+00:00"}}
    ))

    assert reg._is_license_accepted(ORG, NAME) is True
    data = json.loads(reg._ACCEPTANCES_FILE.read_text())
    assert MODEL_REF in data and NAME not in data
    # Idempotent on the migrated file.
    assert reg._is_license_accepted(ORG, NAME) is True
    # A different org does not inherit the migrated acceptance.
    assert reg._is_license_accepted("other-org", "other-model") is False


# ── Server error relay on the download-URL step ───────────────────────


def test_license_login_required_error_is_relayed(monkeypatch, capsys):
    reg._record_license_acceptance(ORG, NAME, LICENSE_ID)  # pass the local gate
    server_msg = "Log in and accept the license on the hub."
    _install_registry_api(monkeypatch, lambda: _FakeResponse(
        {"code": "LICENSE_LOGIN_REQUIRED", "message": server_msg},
        status_code=401,
    ))

    with pytest.raises(SystemExit) as exc:
        reg.cmd_import(_args())
    assert exc.value.code == 1

    out = capsys.readouterr().out
    assert server_msg in out
    assert "logged-in license acceptance" in out


def test_other_structured_error_relayed_with_status(monkeypatch, capsys):
    reg._record_license_acceptance(ORG, NAME, LICENSE_ID)
    _install_registry_api(monkeypatch, lambda: _FakeResponse(
        {"error": "download quota exceeded"},
        status_code=429,
    ))

    with pytest.raises(SystemExit) as exc:
        reg.cmd_import(_args())
    assert exc.value.code == 1

    out = capsys.readouterr().out
    assert "download quota exceeded" in out
    assert "429" in out


def test_non_json_error_body_falls_back_to_raw_print(monkeypatch, capsys):
    reg._record_license_acceptance(ORG, NAME, LICENSE_ID)
    _install_registry_api(monkeypatch, lambda: _FakeResponse(
        status_code=500, json_error=True,
    ))

    with pytest.raises(SystemExit) as exc:
        reg.cmd_import(_args())
    assert exc.value.code == 1

    out = capsys.readouterr().out
    assert "Failed to get download URL:" in out


# ── CLI parser wiring ─────────────────────────────────────────────────


def test_parser_exposes_accept_license_flag():
    from neurobrix.cli import create_parser

    args = create_parser().parse_args(["import", MODEL_REF, "--accept-license"])
    assert args.accept_license is True

    args = create_parser().parse_args(["import", MODEL_REF])
    assert args.accept_license is False
