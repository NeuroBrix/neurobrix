"""`neurobrix import` says the installed neurobrix is behind from the daily update check's cached answer only: the
import opens no connection but the hub's, starts no ask when the answer is fresh, says nothing with the check's switch
on or without a cached answer, and never waits for one.

A local stub hub answers the model record and refuses the download, so the import stops right after the point where
it used to ask the package index itself. Any connection to another address fails the test.
"""
from __future__ import annotations

import contextlib
import http.server
import io
import json
import socket
import subprocess
import sys
import threading
import time
import uuid

import pytest

from neurobrix import __version__
from neurobrix.cli import update_check as uc

SWITCH = "NEUROBRIX_NO_UPDATE_CHECK"


def _stub_hub(name):
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.endswith("/download"):
                status, body = 403, {"message": "download refused by the stub hub", "code": "STUB"}
            else:
                status, body = 200, {"model": {"name": name, "fileSize": 1}}
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A fresh engine home, a stub hub, every connection and every child process recorded; foreign ones fail (every
    client, urllib and requests alike, connects through socket.connect)."""
    monkeypatch.setenv("NEUROBRIX_HOME", str(tmp_path))
    name = f"m{uuid.uuid4().hex[:12]}"
    srv = _stub_hub(name)
    hub = ("127.0.0.1", srv.server_port)
    foreign, children = [], []
    real_connect = socket.socket.connect

    def connect(self, address):
        if tuple(address[:2]) != hub:
            foreign.append(address)
            raise OSError(f"the test forbids a connection to {address}")
        return real_connect(self, address)

    def popen(*a, **kw):
        children.append(a[0] if a else kw.get("args"))
        raise OSError("the test starts no child process")

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(subprocess, "Popen", popen)
    # this neurobrix's version as an install's metadata gives it (a source checkout has none); no nbxtensor read, so
    # only the engine's own line can speak
    monkeypatch.setattr(uc, "installed_version", lambda n: __version__ if n == uc.ENGINE else None)
    yield {"name": name, "url": f"http://{hub[0]}:{hub[1]}", "foreign": foreign, "children": children}
    srv.shutdown()


def _cache(latest_neurobrix, age_s=0.0):
    uc.cache_path().parent.mkdir(parents=True, exist_ok=True)
    uc.cache_path().write_text(json.dumps({"checked_at": time.time() - age_s,
                                           "latest": {"neurobrix": latest_neurobrix}}))


def _import(world, monkeypatch):
    """The command in-process, its streams kept open (the engine's log handler holds the stream it was given)."""
    from neurobrix import cli
    monkeypatch.setattr(sys, "argv", ["neurobrix", "import", f"org/{world['name']}", "--registry", world["url"]])
    out, err = io.StringIO(), io.StringIO()
    started = time.monotonic()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), pytest.raises(SystemExit) as e:
        cli.main()
    return e.value.code, out.getvalue() + err.getvalue(), time.monotonic() - started


def test_an_import_says_it_is_behind_from_a_cached_answer_and_opens_no_connection_but_the_hubs(world, monkeypatch):
    monkeypatch.delenv(SWITCH, raising=False)
    _cache("999.0.0")
    code, said, _ = _import(world, monkeypatch)
    assert code == 1 and "download refused by the stub hub" in said      # the import reached the hub's download
    assert "neurobrix 999.0.0 is published" in said
    assert world["foreign"] == []                                        # no package index, no other host
    assert world["children"] == []                                       # a fresh answer starts no ask


def test_an_import_with_the_update_check_switched_off_says_nothing_and_asks_nothing(world, monkeypatch):
    monkeypatch.setenv(SWITCH, "1")
    _cache("999.0.0", age_s=3 * 24 * 3600)                               # even an old answer: the switch wins
    code, said, _ = _import(world, monkeypatch)
    assert code == 1 and "download refused by the stub hub" in said
    assert "is published" not in said and "999.0.0" not in said
    assert world["foreign"] == [] and world["children"] == []


def test_an_import_without_a_cached_answer_says_nothing_and_does_not_wait(world, monkeypatch):
    monkeypatch.delenv(SWITCH, raising=False)
    assert not uc.cache_path().exists()
    code, said, took = _import(world, monkeypatch)
    assert code == 1 and "download refused by the stub hub" in said
    assert "is published" not in said
    assert world["foreign"] == []                                        # the import itself asks no index
    assert took < uc.TIMEOUT_S                                           # nothing waited for an answer
