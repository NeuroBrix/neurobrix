"""The daemon serves a declared range of protocol versions and refuses a client
outside it at its first frame, by name, before any work (Studio request 7;
supervisor queue item 3, 2026-10-11).

Before: `PROTOCOL_VERSION` was stamped on every response, but a request carried
no version and the daemon checked none — a client of another shape was served
and read answers it could not parse.

A real ServingDaemon on a real Unix socket with the fake engine of
`test_the_daemon_answers_while_it_generates.py` (no model, no GPU).
"""
from __future__ import annotations

import pytest

from neurobrix.serving import protocol
from neurobrix.serving.engine import daemon_identity
from neurobrix.serving.protocol import PROTOCOL_RANGE, PROTOCOL_VERSION, make_request, recv_message, send_message

from test_the_daemon_answers_while_it_generates import _client, _raw, daemon  # noqa: F401  (fixture)

LOW, HIGH = PROTOCOL_RANGE


def _refused_then_closed(sock, request):
    send_message(sock, request)
    reply = recv_message(sock)
    assert "error" in reply and "result" not in reply, reply
    assert recv_message(sock) is None, "the connection stayed open after the refusal"
    return reply["error"]


@pytest.mark.parametrize("version, side", [(HIGH + 1, "daemon"), (LOW - 1, "client")])
def test_a_version_outside_the_range_is_refused_by_name_then_closed(daemon, version, side):
    srv, engine, path = daemon
    error = _refused_then_closed(_raw(path), {"protocol": version, "method": "generate",
                                              "params": {"prompt": "x", "max_tokens": 2}})
    assert f"protocol {version} is not served here" in error
    assert f"accepts protocol {LOW}..{HIGH}" in error and f"update the {side}" in error
    assert engine.log == [], "work ran for a refused client"


@pytest.mark.parametrize("declared", ["1", True, 1.0, None, [1]])
def test_a_version_that_is_not_an_integer_is_refused(daemon, declared):
    _, engine, path = daemon
    error = _refused_then_closed(_raw(path), {"protocol": declared, "method": "status"})
    assert "`protocol` must be an integer" in error


def test_a_client_that_changes_version_mid_connection_is_refused_there(daemon):
    _, _, path = daemon
    s = _raw(path)
    send_message(s, {"protocol": PROTOCOL_VERSION, "method": "status"})
    assert recv_message(s)["result"]["loaded"] is True
    _refused_then_closed(s, {"protocol": HIGH + 1, "method": "status"})


def test_a_request_without_the_field_is_the_unstamped_version(daemon):
    # Clients written before the field (every one until this change) spoke
    # version 1 without saying so; served while that version is in range.
    assert LOW <= protocol.UNSTAMPED_VERSION <= HIGH
    _, _, path = daemon
    s = _raw(path)
    send_message(s, {"method": "status"})
    assert recv_message(s)["result"]["loaded"] is True


def test_the_engines_own_client_stamps_its_version_and_is_served(daemon):
    assert make_request("status")["protocol"] == PROTOCOL_VERSION
    with _client() as c:
        assert c.status()["loaded"] is True


def test_the_range_is_published_where_a_client_reads_before_it_connects():
    assert daemon_identity()["protocol_range"] == [LOW, HIGH]
