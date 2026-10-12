"""A daemon started in the foreground signals its readiness instead of being
polled (item 6 of the engine-side queue; docs/reference/wire-protocol.md,
"Readiness").

A background daemon already signals: through a pipe to the parent on Unix,
through `daemon.ready` on Windows and macOS. In the foreground there was
nothing, so a client that starts the daemon itself (`serve --foreground`)
had to probe `status` until the socket answered. Under `--json` the daemon
now speaks NDJSON on stdout (Studio request 5's form): one `ready` event
with the endpoint, the protocol, the model, the family, the serving strategy
and the pid — or one `error` event with the message — and every human line
goes to stderr. Without `--json` nothing changes on stdout.

What this test does if the code is wrong: the ready cell fails when no
event, more than one line, or an event without the endpoint, the protocol
or the pid reaches stdout (red before the change: the daemon had no event
path at all); the error cell fails if an error does not end as the terminal
`error` event with its message; the silent cell fails if a daemon without
`--json` writes anything to stdout at the signal.

Run: python -m pytest tests/unit/serving/test_the_foreground_daemon_signals_ready.py
"""
from __future__ import annotations

import json
import os

from neurobrix import __version__
from neurobrix.serving.protocol import PROTOCOL_VERSION, endpoint
from neurobrix.serving.server import ServingDaemon


class FakeEngine:
    family = "llm"
    tokenizer = None
    model_name = "fake"
    _warm_serving = True
    _session = None

    def get_status(self):
        return {"model": self.model_name, "loaded": True}


def _daemon(json_events: bool) -> ServingDaemon:
    return ServingDaemon("fake", "none", engine=FakeEngine(), idle_timeout=1, json_events=json_events)


def test_the_ready_signal_is_one_ndjson_event_with_the_endpoint(capsys):
    d = _daemon(True)
    d._signal_ready("ready:Model loaded — fake (llm, warm serving)",
                    ready={"model": "fake", "family": "llm", "strategy": "warm"})
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1, lines
    ev = json.loads(lines[0])
    assert ev["schema"] == "neurobrix.serve/1" and ev["engine"] == __version__
    assert ev["event"] == "ready"
    assert ev["endpoint"] == endpoint() and ev["protocol"] == PROTOCOL_VERSION
    assert ev["model"] == "fake" and ev["family"] == "llm" and ev["strategy"] == "warm"
    assert ev["pid"] == os.getpid()


def test_a_failure_ends_as_the_terminal_error_event(capsys):
    d = _daemon(True)
    d._signal_ready("error:the model has no container")
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1, lines
    ev = json.loads(lines[0])
    assert ev["event"] == "error" and ev["message"] == "the model has no container"
    assert ev["schema"] == "neurobrix.serve/1"


def test_without_json_the_signal_writes_nothing_to_stdout(capsys):
    d = _daemon(False)
    d._signal_ready("ready:Model loaded", ready={"model": "fake", "family": "llm", "strategy": "warm"})
    assert capsys.readouterr().out == ""
