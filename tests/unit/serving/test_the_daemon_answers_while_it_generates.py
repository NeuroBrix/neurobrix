"""The daemon answers while it generates, and acknowledges a cancel only once
the work has stopped (Studio requests 3 and 4; serving/server.py docstring).

A real ServingDaemon serves a real Unix socket, with a fake engine whose
`generate` is a slow decode loop that reads the stop seam and emits per-token
events exactly where both engines do — no model, no GPU.

Injections (each RED, then restored GREEN; see the commit message):
  * a serial accept loop (the connection served on the accept thread) —
    status-during-generation, the concurrent-cancel and vanished-client tests
    time out;
  * the cancel acknowledged before the work stopped (`op.done` set at the
    stop request) — the ordering test fails;
  * the work methods not taking the inference lock — the serialization test
    sees overlapping generations.
  * the listener closed without waking accept() — every test's teardown
    finds the accept loop still alive (a shutdown reply hung the daemon until
    its idle timeout);
  * an orphan recorded as a cancel, and the cancel outcomes collapsed to one
    — the vanished-client and cancel-twice tests fail.
"""
from __future__ import annotations

import json
import shutil
import socket
import struct
import tempfile
import threading
import time
from pathlib import Path

import pytest

from neurobrix import decode_progress, decode_stop
from neurobrix.serving import client as client_mod
from neurobrix.serving.client import DaemonClient
from neurobrix.serving.protocol import MAX_MESSAGE_SIZE, recv_message, send_message
from neurobrix.serving.server import ServingDaemon

pytestmark = pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix-socket transport")

STEP_S = 0.05          # one fake decode step
LONG = 200             # tokens of a generation that runs until stopped (10 s)
STATUS_BOUND_S = 0.5   # status must come back well within one second of a busy daemon


class FakeEngine:
    """Yields tokens slowly through the same seams as both engines' loops."""

    family = "llm"
    tokenizer = None
    model_name = "fake"
    _warm_serving = True
    _session = None

    def __init__(self, sync_s: float = 0.2):
        self.sync_s = sync_s
        self.log = []
        self._log_lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def note(self, what):
        with self._log_lock:
            self.log.append((time.monotonic(), what))

    def generate(self, prompt="", max_tokens=LONG, **kw):
        with self._log_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.note(f"start:{prompt}")
        ids = []
        try:
            for i in range(int(max_tokens)):
                time.sleep(STEP_S)
                ids.append(i + 1)
                done = len(ids) >= int(max_tokens)
                stop = not done and decode_stop.requested(len(ids))
                decode_progress.emit(i, len(ids), i + 1, done or stop)
                if done or stop:
                    break
        finally:
            with self._log_lock:
                self.active -= 1
        self.note(f"returned:{prompt}")
        return {"text": " ".join(map(str, ids)), "tokens": len(ids)}

    def chat(self, message, **kw):
        return self.generate(prompt=message, **kw)["text"]

    def complete(self, messages, tools=None, **kw):
        return self.generate(prompt="complete", **kw)["text"]

    def sync_devices(self):
        time.sleep(self.sync_s)   # a slow sync makes an early ack visible
        self.note("synced")

    def get_status(self):
        return {"model": self.model_name, "loaded": True}

    def chat_template_text(self):
        return "{{ messages }}"

    def new_conversation(self):
        self.note("new_chat")

    def unload(self):
        self.note("unloaded")


@pytest.fixture
def daemon(monkeypatch):
    d = Path(tempfile.mkdtemp(prefix="nbxs"))
    path = d / "d.sock"
    engine = FakeEngine()
    srv = ServingDaemon("fake", "none", engine=engine, idle_timeout=60)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(path))
    sock.listen(srv._settings["listen_backlog"])
    t = threading.Thread(target=srv.serve, args=(sock,), daemon=True)
    t.start()
    monkeypatch.setattr(client_mod, "SOCKET_PATH", path)
    yield srv, engine, path
    srv._ops.stop_all("shutdown")
    srv._close_listener()
    t.join(10)
    shutil.rmtree(d, ignore_errors=True)
    assert not t.is_alive(), "closing the listener did not end the accept loop"


def _client():
    c = DaemonClient()
    c.connect()
    return c


def _raw(path):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(30)
    s.connect(str(path))
    return s


def _frame(payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + payload


def _wait_running(op_id, min_steps=1, timeout=10):
    deadline = time.monotonic() + timeout
    with _client() as c:
        while time.monotonic() < deadline:
            for op in c.status()["operations"]:
                if op["op_id"] == op_id and op["state"] == "running" and op["steps"] >= min_steps:
                    return op
            time.sleep(0.02)
    raise AssertionError(f"{op_id} never ran")


def _background(fn, *a, **kw):
    box = {}

    def run():
        try:
            box["value"] = fn(*a, **kw)
        except Exception as e:   # surfaced by the test through box["error"]
            box["error"] = e
        box["at"] = time.monotonic()
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, box


def _generate(op_id, **kw):
    with _client() as c:
        return c.generate(prompt=op_id, op_id=op_id, **kw)


# ---------------------------------------------------------------- frames
def test_malformed_frames_are_answered_by_name_and_the_connection_survives(daemon):
    _, _, path = daemon
    s = _raw(path)
    for bad, what in ((b"{not json", "not UTF-8 JSON"), (b"[1, 2]", "not an object"),
                      (b"", "Empty frame")):
        s.sendall(_frame(bad))
        reply = recv_message(s)
        assert "malformed frame" in reply["error"] and what in reply["error"]
    send_message(s, {"method": "status", "params": []})
    assert "malformed request" in recv_message(s)["error"]
    send_message(s, {"method": "status"})            # still framed: still served
    assert recv_message(s)["result"]["loaded"] is True
    s.close()


def test_an_invalid_length_is_answered_then_the_connection_is_closed(daemon):
    _, _, path = daemon
    s = _raw(path)
    s.sendall(struct.pack(">I", MAX_MESSAGE_SIZE + 1) + b"x" * 16)
    reply = recv_message(s)
    assert "Message too large" in reply["error"]
    try:                                              # no boundary left to trust:
        closed = recv_message(s) is None              # closed (EOF, or a reset since
    except ConnectionResetError:                      # the unread bytes were dropped)
        closed = True
    assert closed
    s.close()
    with _client() as c:                              # the daemon itself is fine
        assert c.status()["loaded"] is True


def test_a_disconnect_mid_request_leaves_the_daemon_serving(daemon):
    srv, engine, path = daemon
    s = _raw(path)
    s.sendall(struct.pack(">I", 100) + b'{"method": "gen')
    s.close()
    with _client() as c:
        st = c.status()
    assert st["operations"] == [] and not any(w.startswith("start") for _, w in engine.log)


# ------------------------------------------------- answered while it runs
def test_status_is_answered_during_a_generation_within_a_bounded_latency(daemon):
    t, box = _background(_generate, "busy")
    _wait_running("busy")
    with _client() as c:
        t0 = time.monotonic()
        st = c.status()
        latency = time.monotonic() - t0
        assert latency < STATUS_BOUND_S, f"status took {latency:.3f}s during a generation"
        assert [op["op_id"] for op in st["operations"]] == ["busy"]
        assert c.template()["text"] == "{{ messages }}"   # control, no lock
        assert c.cancel("busy")["outcome"] == "cancelled"
    t.join(10)
    assert box["value"]["status"] == "cancelled"


# ------------------------------------------------------------- cancel
def test_a_cancel_is_acknowledged_only_after_the_work_stopped_and_synced(daemon):
    srv, engine, _ = daemon
    t, box = _background(_generate, "c1")
    _wait_running("c1", min_steps=3)
    with _client() as c:
        reply = c.cancel("c1")
        acked = time.monotonic()
    t.join(10)
    when = {w: at for at, w in engine.log}
    assert reply["outcome"] == "cancelled" and reply["state"] == "cancelled"
    assert when["returned:c1"] < when["synced"] < acked, engine.log
    work = box["value"]
    assert work["status"] == "cancelled" and work["op_id"] == "c1"
    assert 3 <= work["steps"] == work["tokens"] < LONG and work["steps"] == reply["steps"]
    assert work["text"].split() == [str(i + 1) for i in range(work["steps"])]


def test_cancel_twice_finished_and_unknown_each_get_a_named_reply(daemon):
    t, box = _background(_generate, "twice")
    _wait_running("twice")
    # two cancels at once: exactly one stopped it
    r1 = _background(lambda: _client().cancel("twice"))
    r2 = _background(lambda: _client().cancel("twice"))
    for th, _ in (r1, r2):
        th.join(30)
    outcomes = sorted(r[1]["value"]["outcome"] for r in (r1, r2))
    assert outcomes == ["already_cancelled", "cancelled"]
    t.join(10)
    with _client() as c:
        assert c.cancel("twice")["outcome"] == "already_cancelled"     # after the fact too
        done = c.generate(prompt="short", op_id="short", max_tokens=2)
        assert done["status"] == "completed" and done["op_id"] == "short"
        late = c.cancel("short")
        assert late["outcome"] == "already_finished" and late["state"] == "completed"
        assert c.cancel("never-was") == {"outcome": "unknown_op", "op_id": "never-was"}
        with pytest.raises(RuntimeError, match="cancel needs params.op_id"):
            c.send("cancel")


def test_a_live_op_id_cannot_be_reused(daemon):
    t, _ = _background(_generate, "dup")
    _wait_running("dup")
    with _client() as c:
        with pytest.raises(RuntimeError, match="names a live operation"):
            c.generate(prompt="x", op_id="dup")
        c.cancel("dup")
    t.join(10)


def test_a_queued_operation_cancelled_never_runs(daemon):
    srv, engine, _ = daemon
    ta, _ = _background(_generate, "first")
    _wait_running("first")
    tb, bb = _background(_generate, "queued")
    deadline = time.monotonic() + 10
    with _client() as c:
        while not any(op["op_id"] == "queued" and op["state"] == "queued"
                      for op in c.status()["operations"]):
            assert time.monotonic() < deadline
            time.sleep(0.02)
        r = c.cancel("queued")
        c.cancel("first")
    tb.join(10); ta.join(10)
    assert r == {"outcome": "cancelled", "op_id": "queued", "state": "cancelled", "steps": 0}
    assert bb["value"]["status"] == "cancelled" and bb["value"]["steps"] == 0
    assert not any(w == "start:queued" for _, w in engine.log)


# ------------------------------------------------------- the inference lock
def test_two_concurrent_generations_are_serialized_by_the_lock(daemon):
    srv, engine, _ = daemon
    ths = [_background(_generate, f"g{i}", max_tokens=6) for i in range(2)]
    for th, _ in ths:
        th.join(30)
    assert all(b["value"]["status"] == "completed" for _, b in ths)
    assert engine.max_active == 1
    spans = {}
    for at, w in engine.log:
        kind, _, name = w.partition(":")
        spans.setdefault(name, {})[kind] = at
    a, b = sorted((spans["g0"], spans["g1"]), key=lambda s: s["start"])
    assert a["returned"] <= b["start"]


# ------------------------------------------------------ vanished clients
def test_a_client_that_vanishes_during_a_generation_orphans_it_never_cancels_it(daemon):
    srv, engine, path = daemon
    s = _raw(path)
    send_message(s, {"method": "generate", "op_id": "gone", "params": {"prompt": "gone"}})
    _wait_running("gone", min_steps=2)
    s.close()
    deadline = time.monotonic() + 10
    with _client() as c:
        while c.status()["operations"]:
            assert time.monotonic() < deadline, "the orphaned generation kept running"
            time.sleep(0.05)
        r = c.cancel("gone")
    assert r["outcome"] == "already_finished" and r["state"] == "orphaned"
    assert 2 <= r["steps"] < LONG


def test_a_streaming_client_that_vanishes_orphans_its_generation(daemon):
    srv, engine, path = daemon
    s = _raw(path)
    send_message(s, {"method": "generate", "op_id": "sgone",
                     "params": {"prompt": "sgone", "stream": True}})
    first = recv_message(s)["stream"]
    assert first["op_id"] == "sgone" and first["step"] == 0
    s.close()
    deadline = time.monotonic() + 10
    with _client() as c:
        while c.status()["operations"]:
            assert time.monotonic() < deadline
            time.sleep(0.05)
        r = c.cancel("sgone")
    assert r["outcome"] == "already_finished" and r["state"] == "orphaned"


# ------------------------------------------------- the existing CLI clients
def test_every_client_method_still_works_and_carries_the_op_id(daemon):
    srv, engine, _ = daemon
    with _client() as c:
        assert c.status()["loaded"] is True
        assert c.template() == {"text": "{{ messages }}"}
        assert c.new_chat() == {"status": "ok"}
        g = c.generate(prompt="p", max_tokens=3)
        assert g["text"] == "1 2 3" and g["status"] == "completed" and g["op_id"]
        events = list(c.generate_stream(prompt="s", max_tokens=2, op_id="st"))
        assert [k for k, _ in events] == ["token", "token", "result"]
        assert all(e["op_id"] == "st" for k, e in events if k == "token")
        assert events[-1][1]["op_id"] == "st" and events[-1][1]["status"] == "completed"
        ch = c.chat("hi", max_tokens=2)
        assert ch["text"] == "1 2" and ch["status"] == "completed"
        cp = c.complete([{"role": "user", "content": "x"}], max_tokens=1)
        assert cp["text"] == "1"
        with pytest.raises(RuntimeError, match="Unknown method"):
            c.send("nope")


def test_shutdown_during_a_generation_stops_it_then_unloads(daemon):
    srv, engine, _ = daemon
    t, box = _background(_generate, "late")
    _wait_running("late")
    with _client() as c:
        assert c.shutdown() == {"status": "ok"}
    t.join(10)
    assert box["value"]["status"] == "stopped"
    when = {w: at for at, w in engine.log}
    assert when["returned:late"] < when["synced"] < when["unloaded"]
