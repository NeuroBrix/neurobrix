"""Behavioral tests for v0.7.0 serve-by-default:
  1. Auto-start daemon when none is running (`neurobrix run` and `neurobrix chat --model`),
     passing `exit_parent=False` so the parent returns cleanly instead of `sys.exit(0)`.
  2. `ServingDaemon.start` / `_daemonize_unix` `exit_parent=False` returns cleanly in parent
     when ready and raises `RuntimeError` when failed (no `SystemExit`), while preserving
     `exit_parent=True` default behavior for explicit `cmd_serve`.
  3. Warm reuse when the daemon already serves the requested model/mode/hardware (zero stop/restart).
  4. Auto-swap with clean VRAM unload (`client.shutdown()` -> daemon exits -> start new daemon)
     when a different model, explicit engine mode, or explicit `--hardware` is requested.
  5. `engines.require_engine` failure refuses BEFORE `stop_daemon`, leaving a healthy daemon running.
  6. `--ephemeral`, `--explain-plan`, `--certified-only`, `NBX_CENSUS=1`,
     `NBX_AUTOTUNE_CERTIFIED_ONLY=1`, and flags not forwarded by `_try_warm_path`
     (`--input-image`, `--input-video`, `--mask-image`, `--video`, `--fps`, `--num-frames`,
     `--system`, `--set`, `--no-resolution-binning`) bypass the warm path so zero flags are dropped.
  7. Explicit `neurobrix serve` compatibility is preserved (refuses if already running).
  8. CLI parser exposes `--ephemeral` on `run` and `--model`/`--hardware`/engine flags on `chat`.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest


class _FakeDaemonState:
    """In-memory state machine simulating a live ServingDaemon + DaemonClient."""

    def __init__(self, initial_model=None, initial_mode="triton", initial_hardware=None, family="llm"):
        self.running = initial_model is not None
        self.model = initial_model
        self.mode = initial_mode
        self.hardware = initial_hardware
        self.family = family
        self.pid = 4242 if self.running else None
        self.events: list[tuple] = []

    def install(self, monkeypatch):
        import neurobrix.serving.client as client_mod
        import neurobrix.serving.server as server_mod
        import neurobrix.core.runtime.engines as engines_mod

        state = self

        class _FakeClient:
            def __init__(self):
                self._sock = None

            @staticmethod
            def is_running():
                return state.running

            @staticmethod
            def get_pid():
                return state.pid if state.running else None

            def connect(self):
                if not state.running:
                    raise RuntimeError("not running")
                state.events.append(("connect", state.model))

            def status(self):
                return {
                    "model": state.model,
                    "mode": state.mode,
                    "hardware": state.hardware,
                    "family": state.family,
                }

            def shutdown(self):
                state.events.append(("shutdown", state.model))
                # Simulates ServingDaemon._shutdown() running self._engine.unload() and exiting cleanly
                state.running = False
                state.model = None
                state.hardware = None
                state.pid = None
                return {"status": "ok"}

            def close(self):
                pass

        class _FakeServingDaemon:
            def __init__(self, model_name, hardware_id=None, mode="triton", idle_timeout=1800.0, http=None):
                self.model_name = model_name
                self.hardware_id = hardware_id
                self.mode = mode
                self.idle_timeout = idle_timeout
                self.http = http

            def start(self, foreground=False, ready_fd=None, ready_file=None, exit_parent=True):
                assert not state.running, "previous daemon must be cleanly stopped before starting a new one"
                state.events.append(("start", self.model_name, self.mode, foreground, exit_parent))
                state.running = True
                state.model = self.model_name
                state.mode = self.mode
                state.hardware = self.hardware_id
                state.pid = 5353

        monkeypatch.setattr(client_mod, "DaemonClient", _FakeClient)
        monkeypatch.setattr(server_mod, "ServingDaemon", _FakeServingDaemon)
        monkeypatch.setattr(engines_mod, "require_engine", lambda mode: None)
        monkeypatch.setattr(engines_mod, "default_mode", lambda: ("triton", "engine: triton (default)"))
        return state


def test_ensure_serving_auto_starts_when_no_daemon_is_running(monkeypatch):
    from neurobrix.cli.commands.serve import ensure_serving

    st = _FakeDaemonState(initial_model=None).install(monkeypatch)
    action = ensure_serving("TinyLlama-1.1B-Chat-v1.0", hardware_id="default-9f169c79", mode="triton")
    assert action == "started"
    assert st.running is True
    assert st.model == "TinyLlama-1.1B-Chat-v1.0"
    assert st.hardware == "default-9f169c79"
    assert st.events == [("start", "TinyLlama-1.1B-Chat-v1.0", "triton", False, False)]


def test_daemonize_unix_exit_parent_flag_controls_parent_exit(monkeypatch):
    import neurobrix.serving.server as server_mod

    daemon = server_mod.ServingDaemon(
        "TinyLlama-1.1B-Chat-v1.0", "default-9f169c79", mode="triton", engine=object()
    )
    served = []
    monkeypatch.setattr(daemon, "_start_serving", lambda: served.append(True))
    monkeypatch.setattr(server_mod, "IS_WINDOWS", False)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "pipe", lambda: (101, 102))
    monkeypatch.setattr(os, "fork", lambda: 9999)  # parent branch
    monkeypatch.setattr(os, "close", lambda fd: None)

    # 1. Ready + exit_parent=False via start() -> returns cleanly, never exits, never calls _start_serving
    monkeypatch.setattr(server_mod, "wait_ready", lambda *a, **kw: {"state": "ready", "pid": 9999})
    monkeypatch.setattr(server_mod, "_announce", lambda rec: True)
    daemon.start(foreground=False, exit_parent=False)
    assert served == []

    # 2. Failed + exit_parent=False via _daemonize_unix() and start() -> raises RuntimeError (no SystemExit)
    monkeypatch.setattr(server_mod, "wait_ready", lambda *a, **kw: {"state": "failed", "error": "oom"})
    monkeypatch.setattr(server_mod, "_announce", lambda rec: False)
    with pytest.raises(RuntimeError, match="Daemon failed to start"):
        daemon._daemonize_unix(exit_parent=False)
    with pytest.raises(RuntimeError, match="Daemon failed to start"):
        daemon.start(foreground=False, exit_parent=False)
    assert served == []

    # 3. Default exit_parent=True preserves sys.exit(0) on ready and sys.exit(1) on failure
    monkeypatch.setattr(server_mod, "wait_ready", lambda *a, **kw: {"state": "ready", "pid": 9999})
    monkeypatch.setattr(server_mod, "_announce", lambda rec: True)
    with pytest.raises(SystemExit) as exc_ok:
        daemon.start(foreground=False)
    assert exc_ok.value.code == 0

    monkeypatch.setattr(server_mod, "wait_ready", lambda *a, **kw: {"state": "failed", "error": "oom"})
    monkeypatch.setattr(server_mod, "_announce", lambda rec: False)
    with pytest.raises(SystemExit) as exc_fail:
        daemon.start(foreground=False)
    assert exc_fail.value.code == 1


def test_ensure_serving_reuses_warm_daemon_without_restart(monkeypatch):
    from neurobrix.cli.commands.serve import ensure_serving

    st = _FakeDaemonState(
        initial_model="TinyLlama-1.1B-Chat-v1.0",
        initial_mode="triton",
        initial_hardware="default-9f169c79",
    ).install(monkeypatch)
    assert ensure_serving("TinyLlama-1.1B-Chat-v1.0") == "reused"
    assert ensure_serving("TinyLlama-1.1B-Chat-v1.0", hardware_id="default-9f169c79") == "reused"
    assert st.running is True
    assert st.model == "TinyLlama-1.1B-Chat-v1.0"
    assert ("shutdown", "TinyLlama-1.1B-Chat-v1.0") not in st.events
    assert not any(ev[0] == "start" for ev in st.events)


def test_ensure_serving_auto_swaps_with_clean_vram_unload(monkeypatch):
    from neurobrix.cli.commands.serve import ensure_serving

    st = _FakeDaemonState(initial_model="TinyLlama-1.1B-Chat-v1.0", initial_mode="triton").install(monkeypatch)
    action = ensure_serving("granite-3.1-1b-a400m-instruct", mode="triton")
    assert action == "swapped"
    assert st.running is True
    assert st.model == "granite-3.1-1b-a400m-instruct"
    # Clean socket shutdown (which runs self._engine.unload()) MUST occur BEFORE new daemon starts
    shutdown_idx = st.events.index(("shutdown", "TinyLlama-1.1B-Chat-v1.0"))
    start_idx = st.events.index(("start", "granite-3.1-1b-a400m-instruct", "triton", False, False))
    assert shutdown_idx < start_idx, st.events


def test_ensure_serving_swaps_when_hardware_differs(monkeypatch):
    from neurobrix.cli.commands.serve import ensure_serving

    st = _FakeDaemonState(
        initial_model="TinyLlama-1.1B-Chat-v1.0",
        initial_mode="triton",
        initial_hardware="hw-profile-a",
    ).install(monkeypatch)
    action = ensure_serving("TinyLlama-1.1B-Chat-v1.0", hardware_id="hw-profile-b", mode="triton")
    assert action == "swapped"
    assert st.running is True
    assert st.hardware == "hw-profile-b"
    shutdown_idx = st.events.index(("shutdown", "TinyLlama-1.1B-Chat-v1.0"))
    start_idx = st.events.index(("start", "TinyLlama-1.1B-Chat-v1.0", "triton", False, False))
    assert shutdown_idx < start_idx, st.events


def test_ensure_serving_validates_engine_before_stopping_running_daemon(monkeypatch):
    import neurobrix.cli.commands.serve as serve_mod
    import neurobrix.core.runtime.engines as engines_mod

    st = _FakeDaemonState(
        initial_model="TinyLlama-1.1B-Chat-v1.0",
        initial_mode="triton",
        initial_hardware="hw-profile-a",
    ).install(monkeypatch)

    def _fail_engine(mode):
        raise RuntimeError(f"engine {mode} unavailable")

    monkeypatch.setattr(engines_mod, "require_engine", _fail_engine)
    stop_calls = []
    monkeypatch.setattr(serve_mod, "stop_daemon", lambda **kw: stop_calls.append(kw) or True)

    with pytest.raises(RuntimeError, match="unavailable"):
        serve_mod.ensure_serving("granite-3.1-1b-a400m-instruct", mode="compiled")

    assert stop_calls == []
    assert st.running is True
    assert st.model == "TinyLlama-1.1B-Chat-v1.0"
    assert ("shutdown", "TinyLlama-1.1B-Chat-v1.0") not in st.events


def test_run_uses_serve_by_default_and_ephemeral_bypasses_daemon(monkeypatch):
    import neurobrix.cli.commands.run as run_mod

    def _args(**kw):
        base = dict(
            model="TinyLlama-1.1B-Chat-v1.0", hardware=None,
            ephemeral=False, explain_plan=False, certified_only=False,
            turns=None, profile_kernels=None, proof_card=None,
            compiled=False, sequential=False, triton=True, triton_sequential=False,
            reference_image=None, reference_audio=None,
            input_image=None, input_video=None, mask_image=None, video=None,
            fps=None, num_frames=None, system=None, set=None, no_resolution_binning=False,
        )
        base.update(kw)
        return SimpleNamespace(**base)

    # 1. Policy check: default is NOT ephemeral; special modes, env vars, and non-warm flags ARE
    for env_key in ("NBX_CENSUS", "NBX_EPHEMERAL", "NBX_AUTOTUNE_CERTIFIED_ONLY"):
        monkeypatch.delenv(env_key, raising=False)
    assert run_mod._should_use_ephemeral(_args()) is False
    assert run_mod._should_use_ephemeral(_args(ephemeral=True)) is True
    assert run_mod._should_use_ephemeral(_args(explain_plan=True)) is True
    assert run_mod._should_use_ephemeral(_args(certified_only=True)) is True
    assert run_mod._should_use_ephemeral(_args(input_image="img.png")) is True
    assert run_mod._should_use_ephemeral(_args(input_video="vid.mp4")) is True
    assert run_mod._should_use_ephemeral(_args(mask_image="mask.png")) is True
    assert run_mod._should_use_ephemeral(_args(video="in.mp4")) is True
    assert run_mod._should_use_ephemeral(_args(fps=24)) is True
    assert run_mod._should_use_ephemeral(_args(num_frames=16)) is True
    assert run_mod._should_use_ephemeral(_args(system="You are helpful.")) is True
    assert run_mod._should_use_ephemeral(_args(set=["global.cfg=7.5"])) is True
    assert run_mod._should_use_ephemeral(_args(no_resolution_binning=True)) is True

    monkeypatch.setenv("NBX_CENSUS", "1")
    assert run_mod._should_use_ephemeral(_args()) is True
    monkeypatch.delenv("NBX_CENSUS", raising=False)

    monkeypatch.setenv("NBX_AUTOTUNE_CERTIFIED_ONLY", "1")
    assert run_mod._should_use_ephemeral(_args()) is True
    monkeypatch.delenv("NBX_AUTOTUNE_CERTIFIED_ONLY", raising=False)

    # 2. End-to-end cmd_run default path: auto-starts daemon via ensure_serving, then executes via _try_warm_path
    st = _FakeDaemonState(initial_model=None).install(monkeypatch)
    warm_calls = []
    monkeypatch.setattr("neurobrix.cli.utils.resolve_model_arg", lambda m: m)
    monkeypatch.setattr(run_mod, "refuse_unread_flags", lambda a: None)
    monkeypatch.setattr(run_mod, "load_turns", lambda a: None)
    monkeypatch.setattr(run_mod, "_try_warm_path", lambda a: (warm_calls.append(a.model), True)[1])

    with pytest.raises(SystemExit) as exc_info:
        run_mod.cmd_run(_args())
    assert exc_info.value.code == 0
    assert st.model == "TinyLlama-1.1B-Chat-v1.0"
    assert ("start", "TinyLlama-1.1B-Chat-v1.0", "triton", False, False) in st.events
    assert warm_calls == ["TinyLlama-1.1B-Chat-v1.0"]

    # 3. End-to-end cmd_run --ephemeral and non-warm flags: NEVER call ensure_serving or _try_warm_path
    for extra_kw in (
        {"ephemeral": True},
        {"input_image": "img.png"},
        {"num_frames": 16},
        {"system": "Be brief."},
        {"set": ["global.cfg=7.5"]},
    ):
        st2 = _FakeDaemonState(initial_model=None).install(monkeypatch)
        ensure_called = []
        monkeypatch.setattr("neurobrix.cli.commands.serve.ensure_serving", lambda *a, **kw: ensure_called.append(a))
        monkeypatch.setattr(run_mod, "find_model", lambda m: (_ for _ in ()).throw(FileNotFoundError("cold-path-reached")))

        with pytest.raises(SystemExit) as exc_ephem:
            run_mod.cmd_run(_args(**extra_kw))
        assert exc_ephem.value.code == 1  # exited at our sentinel FileNotFoundError on the cold path
        assert ensure_called == []
        assert st2.events == []

    # 4. End-to-end cmd_run with NBX_AUTOTUNE_CERTIFIED_ONLY=1: bypasses daemon cleanly
    monkeypatch.setenv("NBX_AUTOTUNE_CERTIFIED_ONLY", "1")
    st3 = _FakeDaemonState(initial_model=None).install(monkeypatch)
    ensure_called = []
    monkeypatch.setattr("neurobrix.cli.commands.serve.ensure_serving", lambda *a, **kw: ensure_called.append(a))
    monkeypatch.setattr(run_mod, "find_model", lambda m: (_ for _ in ()).throw(FileNotFoundError("cold-path-reached")))
    with pytest.raises(SystemExit) as exc_cert:
        run_mod.cmd_run(_args())
    assert exc_cert.value.code == 1
    assert ensure_called == []
    assert st3.events == []


def test_chat_auto_starts_and_auto_swaps_when_model_is_given(monkeypatch):
    import builtins
    import neurobrix.cli.commands.chat as chat_mod

    st = _FakeDaemonState(initial_model="TinyLlama-1.1B-Chat-v1.0", initial_mode="triton", family="llm").install(monkeypatch)
    monkeypatch.setattr("neurobrix.cli.utils.resolve_model_arg", lambda m: m)
    monkeypatch.setattr(chat_mod, "banner", lambda entry: f"{entry} — /quit")
    monkeypatch.setattr(builtins, "input", lambda _prompt="": "/quit")

    args = SimpleNamespace(
        model="granite-3.1-1b-a400m-instruct", hardware=None,
        compiled=False, sequential=False, triton=True, triton_sequential=False,
        max_tokens=None, temperature=None, repetition_penalty=None,
    )
    chat_mod.cmd_chat(args)
    assert st.model == "granite-3.1-1b-a400m-instruct"
    shutdown_idx = st.events.index(("shutdown", "TinyLlama-1.1B-Chat-v1.0"))
    start_idx = st.events.index(("start", "granite-3.1-1b-a400m-instruct", "triton", False, False))
    assert shutdown_idx < start_idx, st.events


def test_explicit_serve_still_refuses_when_daemon_already_running(monkeypatch):
    from neurobrix.cli.commands.serve import cmd_serve

    _FakeDaemonState(initial_model="TinyLlama-1.1B-Chat-v1.0").install(monkeypatch)
    monkeypatch.setattr("neurobrix.cli.utils.resolve_model_arg", lambda m: m)
    args = SimpleNamespace(
        model="granite-3.1-1b-a400m-instruct", hardware=None,
        compiled=False, sequential=False, triton=True, triton_sequential=False,
        timeout=1800, foreground=False, http=False, http_host=None, http_port=None, api_key_file=None,
    )
    with pytest.raises(SystemExit) as exc_info:
        cmd_serve(args)
    assert exc_info.value.code == 1


def test_cli_parser_exposes_ephemeral_on_run_and_model_on_chat():
    from neurobrix.cli import create_parser

    parser = create_parser()
    run_ephem = parser.parse_args(["run", "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "--ephemeral"])
    assert run_ephem.ephemeral is True
    run_default = parser.parse_args(["run", "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"])
    assert run_default.ephemeral is False

    chat_with_model = parser.parse_args(["chat", "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "--triton"])
    assert chat_with_model.model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert chat_with_model.triton is True
    chat_no_model = parser.parse_args(["chat"])
    assert chat_no_model.model is None
