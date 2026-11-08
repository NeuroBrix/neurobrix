"""`neurobrix serve` reads the device only in the process that serves, after the daemon's fork.

2026-11-08 (the clean-terminal inventory, python:3.10-slim + PyPI 0.6.2, one V100): `neurobrix serve --model TinyLlama`
as the README gives it ran the hardware auto-detect in the PARENT before `os.fork()`; the auto-detect reads the device
through the driver, the child inherited an initialised CUDA driver, saw its card as "busy: 32501 MB taken, 0 MB free",
planned cpu_execution and failed every chat turn with "CUDA error: initialization error". With --hardware named, or
--foreground, it served. The parent now passes no profile; the engine resolves it in `load()`, which runs in the child.

What this test does if the code is wrong: the first cell fails when the parent calls the auto-detect (red on main: it
does, line 66-79); the second when the engine loads without resolving an unnamed profile; the third when the
subprocess path (macOS, Windows) passes a profile the parent never resolved.

Run: python -m pytest tests/unit/serving/test_serve_reads_the_device_only_after_the_fork.py
"""
import argparse

import pytest


def test_the_parent_never_reads_the_device_without_a_named_profile(monkeypatch):
    from neurobrix.cli.commands import serve as S
    from neurobrix.core.prism import autodetect
    import neurobrix.serving.server as srv

    def _no(*a, **k):
        raise AssertionError("the serve parent read the device before the fork")
    monkeypatch.setattr(autodetect, "get_or_create_default_profile", _no)
    seen = {}

    class _Daemon:
        def __init__(self, model_name, hardware_id, mode, idle_timeout):
            seen["hardware_id"] = hardware_id

        def start(self, **k):
            seen["started"] = True
    monkeypatch.setattr(srv, "ServingDaemon", _Daemon)
    monkeypatch.setattr(S, "ServingDaemon", _Daemon, raising=False)
    import neurobrix.serving.client as cl
    monkeypatch.setattr(cl.DaemonClient, "is_running", staticmethod(lambda: False))
    args = argparse.Namespace(model="TinyLlama-1.1B-Chat-v1.0", hardware=None, timeout=60, foreground=False,
                              ready_fd=None, ready_file=None, compiled=True, triton=False, triton_sequential=False,
                              sequential=False)
    S.cmd_serve(args)
    assert seen.get("started") and seen["hardware_id"] is None, seen


def test_the_engine_resolves_an_unnamed_profile_where_it_loads(monkeypatch):
    from neurobrix.core.prism import autodetect
    from neurobrix.serving.engine import InferenceEngine
    calls = []
    monkeypatch.setattr(autodetect, "get_or_create_default_profile", lambda: calls.append(1) or "default-test")

    class _Stop(Exception):
        pass

    def _load_profile(hw):
        assert hw == "default-test", hw
        raise _Stop()
    import neurobrix.core.prism as P
    import neurobrix.cli.utils as U
    import neurobrix.nbx as N
    monkeypatch.setattr(P, "load_profile", _load_profile)
    monkeypatch.setattr(U, "find_model", lambda name: __import__("pathlib").Path("/nonexistent"))

    class _C:
        def get_manifest(self):
            return {"family": "llm"}

        def get_neural_components(self):
            return []
    monkeypatch.setattr(N.NBXContainer, "load", staticmethod(lambda p: _C()))
    eng = InferenceEngine("TinyLlama-1.1B-Chat-v1.0", None, "compiled")
    with pytest.raises(_Stop):
        eng.load()
    assert calls == [1] and eng.hardware_id == "default-test"


def test_the_subprocess_path_names_no_profile_the_parent_never_resolved():
    from neurobrix.serving.server import ServingDaemon

    class _E:
        model_name, hardware_id, mode = "TinyLlama-1.1B-Chat-v1.0", None, "compiled"
    d = ServingDaemon.__new__(ServingDaemon)
    d._engine, d._idle_timeout = _E(), 60.0
    assert "--hardware" not in d._foreground_command()
