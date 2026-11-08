"""Every engine call of a daemon runs on ONE long-lived thread (serving/worker.py), whichever connection sent it.

A real ServingDaemon on a real Unix socket with a fake engine that records the thread of each call. Two clients,
two connections: their generate / complete / chat / new_chat, the device syncs and the unload of the shutdown all
land on the same thread, which is none of the connection threads.

Injection (RED, then GREEN; see the commit message): the work run on the connection thread that received it (the
dispatcher before the worker) — two connections give two engine threads and the test fails.
"""
from __future__ import annotations

import shutil
import socket
import tempfile
import threading
from pathlib import Path

import pytest

from neurobrix import decode_progress
from neurobrix.serving import client as client_mod
from neurobrix.serving.client import DaemonClient
from neurobrix.serving.server import ServingDaemon

pytestmark = pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix-socket transport")


class ThreadRecordingEngine:
    family = "llm"
    tokenizer = None
    model_name = "fake"
    _warm_serving = True
    _session = None

    def __init__(self):
        self.threads = []
        self._lock = threading.Lock()

    def _note(self, what):
        with self._lock:
            self.threads.append((what, threading.get_ident()))

    def generate(self, prompt="", max_tokens=3, **kw):
        self._note("generate")
        for i in range(int(max_tokens)):
            decode_progress.emit(i, i + 1, i + 1, i + 1 == int(max_tokens))
        return {"text": "x", "tokens": int(max_tokens)}

    def chat(self, message, **kw):
        self._note("chat")
        return "x"

    def complete(self, messages, tools=None, **kw):
        self._note("complete")
        return "x"

    def sync_devices(self):
        self._note("sync")

    def get_status(self):
        return {"model": self.model_name, "loaded": True}

    def chat_template_text(self):
        return ""

    def new_conversation(self):
        self._note("new_chat")

    def unload(self):
        self._note("unload")


def test_every_engine_call_from_two_connections_runs_on_one_thread(monkeypatch):
    d = Path(tempfile.mkdtemp(prefix="nbxw"))
    path = d / "d.sock"
    engine = ThreadRecordingEngine()
    srv = ServingDaemon("fake", "none", engine=engine, idle_timeout=60)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(path))
    sock.listen(srv._settings["listen_backlog"])
    t = threading.Thread(target=srv.serve, args=(sock,), daemon=True)
    t.start()
    monkeypatch.setattr(client_mod, "SOCKET_PATH", path)
    try:
        a, b = DaemonClient(), DaemonClient()
        a.connect()
        b.connect()
        a.generate("one", max_tokens=2)
        b.complete([{"role": "user", "content": "two"}])
        a.chat("three")
        b.new_chat()
        list(a.generate_stream("four", max_tokens=2))
        b.shutdown()
        a.close()
        b.close()
    finally:
        srv._ops.stop_all("shutdown")
        srv._close_listener()
        t.join(10)
        shutil.rmtree(d, ignore_errors=True)
    kinds = [k for k, _ in engine.threads]
    for needed in ("generate", "complete", "chat", "new_chat", "sync", "unload"):
        assert needed in kinds, f"{needed} never reached the engine: {kinds}"
    idents = {i for _, i in engine.threads}
    assert len(idents) == 1, f"the engine was called from {len(idents)} threads: {engine.threads}"
    assert idents != {t.ident}, "the engine ran on the accept thread"
