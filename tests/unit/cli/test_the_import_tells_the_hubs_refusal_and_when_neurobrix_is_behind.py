"""`neurobrix import` shows the hub's own refusal message, says to upgrade on a 426, and says when it is behind PyPI
(through the update check: its full behaviour is in tests/unit/cli/test_the_engine_says_when_it_is_behind_the_index.py).

A local stub server stands for the hub and for the package index; no network.
"""
from __future__ import annotations

import argparse
import http.server
import json
import threading

import pytest
import requests

from neurobrix import __version__
from neurobrix.cli import update_check as uc
from neurobrix.cli import utils
from neurobrix.cli.commands import registry as reg


def _serve(status, payload):
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


def _refusal(status, payload):
    srv, url = _serve(status, payload)
    try:
        r = requests.get(url, timeout=5)
        with pytest.raises(requests.HTTPError) as e:
            r.raise_for_status()
    finally:
        srv.shutdown()
    return reg.hub_refusal_lines(e.value, "Failed to get download URL")


def test_a_refusal_shows_the_hubs_own_message_and_code():
    # red without the change: hub_refusal_lines does not exist (AttributeError)
    lines = _refusal(403, {"code": "QUOTA", "message": "Daily quota used up"})
    assert lines[0] == "ERROR: Failed to get download URL (403): Daily quota used up"
    assert "   Server error code: QUOTA" in lines
    assert not any(reg.UPGRADE_COMMAND in line for line in lines)


def test_a_426_says_which_command_upgrades():
    # red without the change: no hub_refusal_lines, so no upgrade line on a 426
    lines = _refusal(426, {"message": "Client too old"})
    assert any(reg.UPGRADE_COMMAND in line and __version__ in line for line in lines)


def test_a_refusal_without_a_json_message_still_names_its_status():
    # red without the change: no hub_refusal_lines
    lines = _refusal(500, "plain")
    assert "(500)" in lines[0]


@pytest.fixture
def index_home(tmp_path, monkeypatch):
    """A fresh update-check home, the check switched on, this neurobrix installed and no nbxtensor to compare."""
    monkeypatch.delenv(uc.SWITCH, raising=False)
    monkeypatch.setenv("NEUROBRIX_HOME", str(tmp_path))
    monkeypatch.setattr(uc, "installed_version", lambda name: __version__ if name == uc.ENGINE else None)


def _said_at_start():
    """What a command (the import included) says at start, from the answer the update check stored."""
    return uc.warn(argparse.Namespace(json=False))


@pytest.mark.parametrize("latest, said", [("999.0.0", True), (__version__, False), ("0.0.1", False)])
def test_the_import_says_when_the_installed_release_is_behind_the_index(index_home, monkeypatch, latest, said):
    # the real HTTP ask against a stub index, then the line the import prints from the stored answer
    srv, url = _serve(200, {"info": {"version": latest}})
    monkeypatch.setattr(utils, "PYPI_INDEX_URL", url)
    try:
        uc.refresh()
    finally:
        srv.shutdown()
    lines = _said_at_start()
    if said:
        assert any(latest in line and __version__ in line and uc.update_command(uc.ENGINE) in line for line in lines)
    else:
        assert lines == []


def test_an_index_that_does_not_answer_is_silence(index_home, monkeypatch):
    monkeypatch.setattr(utils, "PYPI_INDEX_URL", "http://127.0.0.1:1")
    uc.refresh()
    assert uc.read_cache()["latest"] == {}
    assert _said_at_start() == []
