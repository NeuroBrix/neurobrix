"""The daemon refuses a client whose protocol is outside its range, at the
first frame, with a message that names both numbers and the engine version
(Studio request 7; docs/reference/studio-engine-requests.md, item 4 of the
engine-side queue).

A request may carry `protocol`, the client's wire version. One outside
`[PROTOCOL_MIN, PROTOCOL_VERSION]` is answered with an error and the
connection is closed: nothing further from that client can be trusted. A
request without the field is a client from before the handshake and is
accepted as protocol 1 — said in the protocol module so no one reads the
absence as a guess. The CLI client stamps its own version on every request.

What this test does if the code is wrong: before the range existed every
cell here but the two acceptance cells was red — a request stamped protocol
0, 99, "1", 1.5 or true was served like any other and the connection stayed
open (seen red on 6ce86b39, 2026-10-13); if the refusal stops naming the
client's number, the daemon's range or the engine version, the message cell
fails; if a refusal stops closing the connection, the closing cell fails; if
the CLI client stops stamping its version, or a bare request is no longer
accepted, the acceptance cells fail.

Run: python -m pytest tests/unit/serving/test_the_daemon_refuses_a_protocol_outside_its_range.py
"""
from __future__ import annotations

import shutil
import socket
import tempfile
import threading
from pathlib import Path

import pytest

from neurobrix import __version__
from neurobrix.serving import client as client_mod
from neurobrix.serving.client import DaemonClient
from neurobrix.serving.protocol import (
    PROTOCOL_MIN,
    PROTOCOL_VERSION,
    make_request,
    recv_message,
    send_message,
)
from neurobrix.serving.server import ServingDaemon

pytestmark = pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix-socket transport")


class FakeEngine:
    """Only what `status` reads: no model, no GPU."""

    family = "llm"
    tokenizer = None
    model_name = "fake"
    _warm_serving = True
    _session = None

    def get_status(self):
        return {"model": self.model_name, "loaded": True}

    def sync_devices(self):
        pass

    def unload(self):
        pass


@pytest.fixture
def daemon(monkeypatch):
    d = Path(tempfile.mkdtemp(prefix="nbxp"))
    path = d / "d.sock"
    srv = ServingDaemon("fake", "none", engine=FakeEngine(), idle_timeout=60)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(path))
    sock.listen(srv._settings["listen_backlog"])
    t = threading.Thread(target=srv.serve, args=(sock,), daemon=True)
    t.start()
    monkeypatch.setattr(client_mod, "SOCKET_PATH", path)
    yield srv, path
    srv._ops.stop_all("shutdown")
    srv._close_listener()
    t.join(10)
    shutil.rmtree(d, ignore_errors=True)
    assert not t.is_alive(), "closing the listener did not end the accept loop"


def _raw(path):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(5)
    s.connect(str(path))
    return s


def _status(protocol):
    req = make_request("status")
    if protocol is _ABSENT:
        req.pop("protocol", None)
    else:
        req["protocol"] = protocol
    return req


_ABSENT = object()


@pytest.mark.parametrize("protocol", [PROTOCOL_MIN - 1, PROTOCOL_VERSION + 1, 99, "1", 1.5, True, None])
def test_a_protocol_outside_the_range_is_refused_by_name_and_the_connection_closed(daemon, protocol):
    _, path = daemon
    s = _raw(path)
    send_message(s, _status(protocol))
    reply = recv_message(s)
    assert reply is not None and "error" in reply, reply
    msg = reply["error"]
    assert str(protocol) in msg, msg                                  # the client's number
    assert f"{PROTOCOL_MIN}" in msg and f"{PROTOCOL_VERSION}" in msg, msg   # the daemon's range
    assert __version__ in msg, msg                                    # the engine's version
    assert reply["protocol"] == PROTOCOL_VERSION and reply["engine"] == __version__
    assert recv_message(s) is None, "the connection stayed open after a refused protocol"
    s.close()


def test_a_request_without_the_field_is_a_pre_handshake_client_accepted_as_protocol_one(daemon):
    _, path = daemon
    s = _raw(path)
    send_message(s, _status(_ABSENT))
    reply = recv_message(s)
    assert reply is not None and "result" in reply, reply
    assert reply["result"]["loaded"] is True
    s.close()


@pytest.mark.parametrize("protocol", sorted({PROTOCOL_MIN, PROTOCOL_VERSION}))
def test_every_version_inside_the_range_is_served(daemon, protocol):
    _, path = daemon
    s = _raw(path)
    send_message(s, _status(protocol))
    reply = recv_message(s)
    assert reply is not None and "result" in reply, reply
    s.close()


def test_the_cli_client_stamps_its_own_protocol_and_is_served(daemon):
    assert make_request("status")["protocol"] == PROTOCOL_VERSION
    c = DaemonClient()
    c.connect()
    try:
        assert c.status()["loaded"] is True
    finally:
        c.close()


def test_a_refusal_leaves_the_daemon_serving_other_clients(daemon):
    _, path = daemon
    bad = _raw(path)
    send_message(bad, _status(PROTOCOL_VERSION + 1))
    assert "error" in recv_message(bad)
    bad.close()
    good = _raw(path)
    send_message(good, _status(PROTOCOL_VERSION))
    reply = recv_message(good)
    assert reply is not None and "result" in reply, reply
    good.close()
