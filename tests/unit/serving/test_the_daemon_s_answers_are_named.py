"""Three findings of the Mac's point-8 validation on MPS (2026-11-09 20:34): (1) after the daemon serving the asked model
answered an error, `neurobrix run` attempted a cold run (refused, saying a daemon serves "another model"): the daemon's
answer is the run's answer; (2) the daemon named its cache entry where a user reads the model (serve's line, chat's
status): the one name is shown; (3) a socket path longer than AF_UNIX holds died with Python's raw OSError.

What this test would do if the code were wrong: a run that goes on after the daemon's error (red), the status without
the one name (red), a long socket path raising the raw OSError (red)."""
from __future__ import annotations

import json
import socket
import types

import pytest


def test_the_daemon_s_error_ends_the_run(monkeypatch):
    from neurobrix.cli.commands import run as R
    from neurobrix.serving import client as C

    class _Client:
        is_running = staticmethod(lambda: True)
        get_pid = staticmethod(lambda: 1)

        def connect(self):
            pass

        def status(self):
            return {"model": "m", "family": "llm", "state": "ready"}

        def close(self):
            pass

        def generate(self, **kw):
            raise RuntimeError("out of memory, refused by name")
    monkeypatch.setattr(C, "DaemonClient", _Client)
    args = types.SimpleNamespace(model="m", prompt="p", output=None, mode=None, speaker=None, seed=None, audio=None,
                                 steps=None, cfg=None, temperature=None, repetition_penalty=None, top_k=None,
                                 top_p=None, max_tokens=None, chat_mode=None, height=None, width=None)
    with pytest.raises(SystemExit) as stop:
        R._try_warm_path(args)
    assert stop.value.code == 1


def test_the_status_carries_the_one_name(tmp_path, monkeypatch):
    from neurobrix.core import paths
    from neurobrix.serving.engine import InferenceEngine
    entry = tmp_path / "Sana_1600M_1024px_MultiLing"
    entry.mkdir()
    (entry / "manifest.json").write_text("{}")
    (entry / "name.json").write_text(json.dumps({"name": "Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers"}))
    monkeypatch.setattr(paths, "cache_dir", lambda: tmp_path)
    e = InferenceEngine.__new__(InferenceEngine)
    e.model_name = entry.name
    assert e.one_name() == "Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers"


def test_a_socket_path_too_long_is_refused_by_name(tmp_path):
    from neurobrix.serving.server import _bind
    path = tmp_path / ("d" * 200) / "daemon.sock"
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match="NEUROBRIX_HOME"):
            _bind(s, str(path))
    finally:
        s.close()
