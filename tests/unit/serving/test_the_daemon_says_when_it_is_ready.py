"""The daemon says when it is ready — bound, listening, model loaded — or that it
failed, as one record its launcher reads without polling (supervisor queue
item 6, 2026-10-11).

Before: the Linux background path read a pipe; the macOS and Windows background
paths polled a `daemon.ready` file every 0.5 s with a 300 s deadline written in
the code, the file was shared by every instance (two named daemons starting
together read each other's signal), a failed write was swallowed, and a
foreground daemon said nothing a launcher could read (macOS even left a
`daemon.ready` behind that nobody removed).

Now: every run prints one line `NBX-READY <json record>` on its stdout once
the listener accepts; a launcher may also pass `--ready-fd N` (POSIX: the same
record, one line, on that inherited pipe, then the fd is closed) or
`--ready-file PATH` (the record, written whole). `wait_ready` is the launcher
side: it blocks on the pipe (no sleep, no deadline: a child that dies closes
the pipe), and on Windows, where no fd is inherited, it watches the instance's
own ready file at `serving.ready_poll_s`.
"""
from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pytest

from neurobrix.serving import protocol, server
from neurobrix.serving.protocol import PROTOCOL_RANGE, recv_message, send_message
from neurobrix.serving.server import READY_LINE, ServingDaemon, wait_ready

from test_the_daemon_answers_while_it_generates import FakeEngine, _raw


class LoadingEngine(FakeEngine):
    def __init__(self, fail: str | None = None):
        super().__init__()
        self.fail = fail

    def load(self):
        if self.fail:
            raise RuntimeError(self.fail)

    def unload(self):
        pass

    def get_status(self):
        return {"model": self.model_name, "loaded": True}


@pytest.fixture
def instance(monkeypatch):
    """A named instance in a scratch directory, seen by the server and by
    `protocol.endpoint()` alike; signal handlers left alone."""
    d = Path(tempfile.mkdtemp(prefix="nbxr"))
    inst = protocol._configure(sys.platform, {"NBX_SOCKET_PATH": str(d / "r.sock")})
    for mod in (server, protocol):
        monkeypatch.setattr(mod, "INSTANCE", inst)
    monkeypatch.setattr(server, "SOCKET_PATH", inst.socket_path)
    monkeypatch.setattr(server, "PID_PATH", inst.pid_path)
    monkeypatch.setattr(server.signal, "signal", lambda *a: None)
    return inst


def _ready_lines(out: str):
    return [json.loads(l[len(READY_LINE) + 1:]) for l in out.splitlines() if l.startswith(READY_LINE + " ")]


def test_ready_is_said_once_the_listener_accepts_on_stdout_and_on_the_pipe(instance, capsys):
    r, w = os.pipe()
    seen = {}

    def launcher():
        with os.fdopen(r) as f:
            seen["record"] = json.loads(f.readline())
            seen["eof"] = f.read() == ""
        s = _raw(instance.socket_path)          # ready means a client is served now
        send_message(s, {"method": "status"})
        seen["status"] = recv_message(s)["result"]
        send_message(s, {"method": "shutdown"})
        recv_message(s)

    t = threading.Thread(target=launcher)
    t.start()
    ServingDaemon("fake", "none", engine=LoadingEngine(), idle_timeout=60).start(foreground=True, ready_fd=w)
    t.join(30)

    rec = seen["record"]
    assert rec["state"] == "ready" and rec["pid"] == os.getpid()
    assert rec["model"] == "fake" and rec["family"] == "llm" and rec["serving"] == "warm"
    assert rec["protocol_range"] == list(PROTOCOL_RANGE)
    assert rec["endpoint"] == {"kind": "unix", "path": str(instance.socket_path)}
    assert seen["eof"], "the ready fd stayed open after the record"
    assert seen["status"]["loaded"] is True
    assert _ready_lines(capsys.readouterr().out) == [rec], "stdout must carry the same record, once"
    assert not instance.ready_path.exists(), "a foreground run without --ready-file wrote a ready file"


def test_a_load_failure_is_a_failed_record_then_the_error(instance, capsys):
    r, w = os.pipe()
    with pytest.raises(RuntimeError, match="no weights here"):
        ServingDaemon("fake", "none", engine=LoadingEngine(fail="no weights here"),
                      idle_timeout=60).start(foreground=True, ready_fd=w)
    assert select.select([r], [], [], 5)[0], "the launcher's pipe got no record and stayed open"
    with os.fdopen(r) as f:
        rec = json.loads(f.readline())
    assert rec["state"] == "failed" and "no weights here" in rec["error"]
    assert _ready_lines(capsys.readouterr().out) == [rec]


def test_an_undeliverable_ready_signal_is_said_not_swallowed(instance, capsys):
    r, w = os.pipe()
    os.close(r)                                  # the launcher is gone
    d = ServingDaemon("fake", "none", engine=LoadingEngine(), idle_timeout=60)
    d._ready_fd = w
    d._signal_ready(d._ready_record())
    assert f"ready signal not delivered on fd {w}" in capsys.readouterr().out


def test_the_ready_file_is_written_whole_and_belongs_to_the_instance(instance):
    assert instance.ready_path == instance.socket_path.with_suffix(".ready")
    default = protocol._configure(sys.platform, {})
    assert default.ready_path == protocol.DAEMON_DIR / "daemon.ready"
    d = ServingDaemon("fake", "none", engine=LoadingEngine(), idle_timeout=60)
    d._ready_file = instance.ready_path
    d._signal_ready(d._ready_record())
    assert json.loads(instance.ready_path.read_text())["state"] == "ready"
    assert not list(instance.ready_path.parent.glob("*.tmp")), "a partial ready file was left"


_CHILD = """
import json, os, sys, time
time.sleep(0.3)
rec = json.dumps({"state": "ready", "pid": os.getpid()}) + "\\n"
if sys.argv[1] == "fd":
    os.write(int(sys.argv[2]), rec.encode())
else:
    open(sys.argv[2] + ".tmp", "w").write(rec); os.replace(sys.argv[2] + ".tmp", sys.argv[2])
time.sleep(0.5)
"""


def test_the_launcher_blocks_on_the_pipe_and_never_sleeps(monkeypatch):
    def _no_sleep(s):
        raise AssertionError(f"the POSIX launcher polled (sleep {s})")
    monkeypatch.setattr(server.time, "sleep", _no_sleep)
    r, w = os.pipe()
    proc = subprocess.Popen([sys.executable, "-c", _CHILD, "fd", str(w)], pass_fds=(w,))
    os.close(w)
    rec = wait_ready(proc, read_fd=r)
    assert rec == {"state": "ready", "pid": proc.pid}
    proc.wait()


def test_a_child_that_dies_before_ready_is_named_with_its_exit_code():
    r, w = os.pipe()
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(7)"], pass_fds=(w,))
    os.close(w)
    rec = wait_ready(proc, read_fd=r)
    assert rec["state"] == "failed" and "exited with code 7 before it was ready" in rec["error"]


def test_the_windows_launcher_reads_the_instances_file(tmp_path):
    ready = tmp_path / "w.ready"
    proc = subprocess.Popen([sys.executable, "-c", _CHILD, "file", str(ready)])
    rec = wait_ready(proc, ready_file=ready)
    assert rec == {"state": "ready", "pid": proc.pid}
    assert not ready.exists(), "the launcher left the ready file behind"
    proc.wait()


def test_the_windows_poll_interval_is_a_setting():
    assert server.serving_settings()["ready_poll_s"] > 0
