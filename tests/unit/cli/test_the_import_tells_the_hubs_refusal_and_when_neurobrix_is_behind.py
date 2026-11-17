"""`neurobrix import` shows the hub's own refusal message and says to upgrade on a 426 (when it is behind
PyPI: tests/unit/cli/test_the_engine_says_when_it_is_behind_the_index.py).

A local stub server stands for the hub; no network.
"""
from __future__ import annotations

import http.server
import json
import threading

import pytest
import requests

from neurobrix import __version__
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
