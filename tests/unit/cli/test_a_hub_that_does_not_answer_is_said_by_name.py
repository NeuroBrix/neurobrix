"""When the hub does not answer, `hub` and `import` say so by name: which hub, what to do, exit 1, no traceback.

Four classes: a refused connection, a timeout, a 5xx and a body that is not JSON. A local stub server
stands for the hub; no network.
"""
from __future__ import annotations

import argparse
import http.server
import logging
import socket
import threading

import pytest
import requests

from neurobrix.cli.commands import registry as reg


def _serve(status, body):
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(status)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


def _closed_port_url():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


def _doors(registry):
    return [
        (reg.cmd_hub, argparse.Namespace(registry=registry, category=None, search=None, json=False)),
        (reg.cmd_import, argparse.Namespace(model_ref="org/name", registry=registry, force=False,
                                            no_keep=False, accept_license=True, json=False)),
    ]


def _assert_named(registry, caplog, tmp_path, monkeypatch):
    monkeypatch.setattr(reg, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(reg, "STORE_DIR", tmp_path / "store")
    for door, args in _doors(registry):
        caplog.clear()
        with caplog.at_level(logging.ERROR):
            with pytest.raises(SystemExit) as e:
                door(args)
        assert e.value.code not in (0, None)
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert f"The hub at {registry} did not answer" in text, text
        assert "Retry" in text and "--registry" in text


def test_a_hub_that_refuses_the_connection_is_said_by_name(caplog, tmp_path, monkeypatch):
    _assert_named(_closed_port_url(), caplog, tmp_path, monkeypatch)


def test_a_hub_that_times_out_is_said_by_name(caplog, tmp_path, monkeypatch):
    def slow(*a, **k):
        raise requests.ReadTimeout("read timed out")
    monkeypatch.setattr(requests, "get", slow)
    _assert_named("http://hub.invalid", caplog, tmp_path, monkeypatch)


def test_a_hub_that_answers_5xx_is_said_by_name(caplog, tmp_path, monkeypatch):
    srv, url = _serve(503, b"<html>down</html>")
    try:
        _assert_named(url, caplog, tmp_path, monkeypatch)
    finally:
        srv.shutdown()


def test_a_hub_that_answers_a_body_that_is_not_json_is_said_by_name(caplog, tmp_path, monkeypatch):
    srv, url = _serve(200, b"<html>maintenance</html>")
    try:
        _assert_named(url, caplog, tmp_path, monkeypatch)
    finally:
        srv.shutdown()
