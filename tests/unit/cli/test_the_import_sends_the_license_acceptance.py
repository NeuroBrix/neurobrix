"""`neurobrix import` asks the hub for a download with the licence acceptance when the model carries a licence (the hub
answers 401 without it, 2026-11-01), and without any header when the model is free (Hocine 2026-11-01 21:26: no licence
= free). What this test would do if the code were wrong: the header dropped (a licensed import refused by the hub) or
sent for a free model (an acceptance nobody gave, recorded by the hub) — RED."""
from __future__ import annotations

import types

import pytest

from neurobrix.cli.commands import registry as R


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


class _Stop(Exception):
    pass


def _import(monkeypatch, tmp_path, license_id):
    import requests
    seen = {}

    def get(url, timeout=None, headers=None, **kw):
        if url.endswith("/download"):
            seen["headers"] = dict(headers or {})
            raise _Stop()
        return _Resp({"model": {"license": license_id, "fileSize": 0}})

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(R, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(R, "_is_license_accepted", lambda org, name: True)
    args = types.SimpleNamespace(registry="http://hub.invalid", model_ref="org/model", force=False, json=False,
                                 accept_license=True, no_keep=False)
    with pytest.raises(_Stop):
        R._import_body(args)
    return seen["headers"]


def test_a_licensed_model_is_downloaded_with_the_acceptance(monkeypatch, tmp_path):
    assert _import(monkeypatch, tmp_path, "apache-2.0").get("X-License-Accepted") == "true"


def test_a_free_model_is_downloaded_with_no_acceptance(monkeypatch, tmp_path):
    assert "X-License-Accepted" not in _import(monkeypatch, tmp_path, None)
