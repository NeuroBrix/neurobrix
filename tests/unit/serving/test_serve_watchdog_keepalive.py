"""Unit tests for P1.3 / P4 Lot 1: GPU sticky-fault watchdog & --keep-alive idle VRAM auto-unload."""

from __future__ import annotations

import logging
import socket
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from neurobrix.cli import create_parser
from neurobrix.serving.client import DaemonClient
from neurobrix.serving.fault_watchdog import abort_on_sticky_fault, is_sticky_gpu_fault
from neurobrix.serving.server import ServingDaemon, validate_keep_alive


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeEngine:
    """Minimal CPU-only engine stub tracking load/unload/generate/sync calls."""

    def __init__(self, *, fail_exc: Optional[BaseException] = None,
                 sync_exc: Optional[BaseException] = None) -> None:
        self.model_name = "fake-model"
        self.hardware_id = "cpu-test"
        self.mode = "triton"
        self.family = "llm"
        self.max_context = None
        self._warm_serving = True
        self._session = None
        self._is_loaded = True
        self._fail_exc = fail_exc
        self._sync_exc = sync_exc
        self.load_calls = 0
        self.unload_calls = 0
        self.generate_calls = 0
        self.sync_calls = 0

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self) -> None:
        self.load_calls += 1
        self._is_loaded = True

    def unload(self) -> None:
        self.unload_calls += 1
        self._is_loaded = False

    def sync_devices(self) -> None:
        self.sync_calls += 1
        if self._sync_exc is not None:
            raise self._sync_exc

    def generate(self, **params) -> Dict[str, Any]:
        self.generate_calls += 1
        if self._fail_exc is not None:
            raise self._fail_exc
        return {"text": f"reply-{self.generate_calls}"}

    def get_status(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "hardware": self.hardware_id,
            "family": self.family,
            "loaded": self._is_loaded,
            "mode": self.mode,
        }


def _start_daemon_on_unix_socket(tmp_path: Path, daemon: ServingDaemon):
    sock_path = tmp_path / "daemon.sock"
    srv_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv_sock.bind(str(sock_path))
    srv_sock.listen(8)
    thread = threading.Thread(target=daemon.serve, args=(srv_sock,), daemon=True)
    thread.start()
    return sock_path, thread


def _connect(sock_path: Path) -> DaemonClient:
    c = DaemonClient()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(str(sock_path))
    c._sock = sock
    return c


# ---------------------------------------------------------------------------
# 1. Sticky GPU fault detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exc", [
    RuntimeError("CUDA error: an illegal memory access was encountered"),
    RuntimeError("CUDA error: device-side assert triggered"),
    RuntimeError("CUDA error: unspecified launch failure"),
    RuntimeError("CUDA error: misaligned address"),
    RuntimeError("CUDA error: uncorrectable ECC error encountered"),
    RuntimeError("CUDA context is destroyed (cudaErrorContextIsDestroyed)"),
    RuntimeError("HIP error: hipErrorIllegalAddress in kernel"),
    RuntimeError("Memory access fault by GPU node-1"),
    RuntimeError("MTLCommandBufferStatusError: command buffer execution failed"),
    RuntimeError("Metal command buffer execution failed (GPU hang)"),
])
def test_is_sticky_gpu_fault_detects_unrecoverable_gpu_errors(exc):
    assert is_sticky_gpu_fault(exc) is True


def test_is_sticky_gpu_fault_walks_exception_chain():
    root = RuntimeError("CUDA error: an illegal memory access was encountered")
    outer = RuntimeError("decode step failed")
    outer.__cause__ = root
    assert is_sticky_gpu_fault(outer) is True


@pytest.mark.parametrize("exc", [
    ValueError("invalid prompt"),
    KeyError("missing_key"),
    RuntimeError("MissingRuntimeValue: input_ids not provided"),
    RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
    RuntimeError("cudaErrorMemoryAllocation: out of memory"),
    RuntimeError("HIP out of memory while allocating arena"),
    RuntimeError("DeviceOutOfMemory: allocation exceeded budget"),
])
def test_is_sticky_gpu_fault_ignores_non_sticky_and_oom_errors(exc):
    assert is_sticky_gpu_fault(exc) is False


# ---------------------------------------------------------------------------
# 2. abort_on_sticky_fault behavior & ServingDaemon wiring
# ---------------------------------------------------------------------------

def test_abort_on_sticky_fault_runs_cleanup_logs_and_calls_exit_fn(tmp_path, caplog):
    sock_file = tmp_path / "daemon.sock"
    pid_file = tmp_path / "daemon.pid"
    sock_file.write_text("sock")
    pid_file.write_text("12345")

    exit_codes: List[int] = []

    def cleanup():
        sock_file.unlink()
        pid_file.unlink()

    with caplog.at_level(logging.ERROR, logger="neurobrix.serving.fault_watchdog"):
        triggered = abort_on_sticky_fault(
            RuntimeError("CUDA error: an illegal memory access was encountered"),
            cleanup_fn=cleanup,
            exit_fn=exit_codes.append,
        )

    assert triggered is True
    assert exit_codes == [1]
    assert not sock_file.exists()
    assert not pid_file.exists()
    assert any("[Watchdog]" in rec.message for rec in caplog.records)


def test_abort_on_sticky_fault_calls_exit_even_if_cleanup_fails():
    exit_codes: List[int] = []

    def bad_cleanup():
        raise OSError("disk read-only")

    triggered = abort_on_sticky_fault(
        RuntimeError("MTLCommandBufferStatusError: command buffer execution failed"),
        cleanup_fn=bad_cleanup,
        exit_fn=exit_codes.append,
    )
    assert triggered is True
    assert exit_codes == [1]


def test_abort_on_sticky_fault_noop_on_recoverable_error():
    called = []
    exit_codes: List[int] = []
    triggered = abort_on_sticky_fault(
        RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB"),
        cleanup_fn=lambda: called.append("cleanup"),
        exit_fn=exit_codes.append,
    )
    assert triggered is False
    assert called == []
    assert exit_codes == []


def test_daemon_run_operation_aborts_on_sticky_fault_and_cleans_files(tmp_path, monkeypatch):
    sock_file = tmp_path / "nbx.sock"
    pid_file = tmp_path / "nbx.pid"
    ready_file = tmp_path / "nbx.ready"
    for f in (sock_file, pid_file, ready_file):
        f.write_text("x")

    import neurobrix.serving.server as srv_mod
    monkeypatch.setattr(srv_mod.INSTANCE, "instance_files", lambda: [sock_file, pid_file])

    exit_codes: List[int] = []
    engine = _FakeEngine(fail_exc=RuntimeError("CUDA error: device-side assert triggered"))
    daemon = ServingDaemon(
        "fake-model", "cpu-test", mode="triton", idle_timeout=60.0,
        engine=engine, watchdog_exit_fn=exit_codes.append,
    )
    daemon._ready_file = ready_file
    op = daemon._ops.open("generate", None)
    daemon.run_operation(op, lambda: engine.generate(prompt="hi"))

    assert exit_codes == [1]
    assert not sock_file.exists()
    assert not pid_file.exists()
    assert not ready_file.exists()


def test_daemon_run_operation_aborts_on_sticky_fault_in_sync_devices():
    exit_codes: List[int] = []
    engine = _FakeEngine(sync_exc=RuntimeError("CUDA error: an illegal memory access was encountered"))
    daemon = ServingDaemon(
        "fake-model", "cpu-test", mode="triton", idle_timeout=60.0,
        engine=engine, watchdog_exit_fn=exit_codes.append,
    )
    op = daemon._ops.open("generate", None)
    daemon.run_operation(op, lambda: engine.generate(prompt="hi"))
    assert exit_codes == [1]


def test_daemon_run_operation_does_not_abort_on_oom_or_value_error():
    for exc in (ValueError("bad input"), RuntimeError("CUDA out of memory")):
        exit_codes: List[int] = []
        engine = _FakeEngine(fail_exc=exc)
        daemon = ServingDaemon(
            "fake-model", "cpu-test", mode="triton", idle_timeout=60.0,
            engine=engine, watchdog_exit_fn=exit_codes.append,
        )
        op = daemon._ops.open("generate", None)
        state, _ = daemon.run_operation(op, lambda: engine.generate(prompt="hi"))
        assert state == "failed"
        assert exit_codes == []


# ---------------------------------------------------------------------------
# 3. --keep-alive CLI parsing & validation
# ---------------------------------------------------------------------------

def test_cli_parse_keep_alive():
    parser = create_parser()
    assert parser.parse_args(["serve", "--model", "m"]).keep_alive is None
    assert parser.parse_args(["serve", "--model", "m", "--keep-alive", "300"]).keep_alive == 300.0
    assert parser.parse_args(["serve", "--model", "m", "--keep-alive", "0"]).keep_alive == 0.0
    assert parser.parse_args(["serve", "--model", "m", "--keep-alive", "-1"]).keep_alive == -1.0


@pytest.mark.parametrize("raw,expected", [
    (None, None),
    (-1, -1.0),
    (-1.0, -1.0),
    (0, 0.0),
    (0.0, 0.0),
    (300, 300.0),
    (0.25, 0.25),
])
def test_validate_keep_alive_accepts_valid_values(raw, expected):
    assert validate_keep_alive(raw) == expected


@pytest.mark.parametrize("bad", [-2, -0.5, -100.0, float("nan"), float("inf"), float("-inf"), True, False, "abc"])
def test_validate_keep_alive_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        validate_keep_alive(bad)


def test_foreground_command_forwards_keep_alive():
    engine = _FakeEngine()
    d_with = ServingDaemon("fake-model", "cpu-test", mode="triton", idle_timeout=60.0,
                           engine=engine, keep_alive=120.0)
    cmd = d_with._foreground_command()
    assert "--keep-alive" in cmd
    assert cmd[cmd.index("--keep-alive") + 1] == "120.0"

    d_without = ServingDaemon("fake-model", "cpu-test", mode="triton", idle_timeout=60.0,
                              engine=engine)
    assert "--keep-alive" not in d_without._foreground_command()


# ---------------------------------------------------------------------------
# 4. --keep-alive idle TTL unload + transparent reload & keep_alive=0
# ---------------------------------------------------------------------------

def test_keep_alive_idle_ttl_unloads_and_reloads_transparently(tmp_path):
    engine = _FakeEngine()
    daemon = ServingDaemon(
        "fake-model", "cpu-test", mode="triton", idle_timeout=5.0,
        engine=engine, keep_alive=0.08,
    )
    sock_path, thread = _start_daemon_on_unix_socket(tmp_path, daemon)
    try:
        # 1st request: served warm
        c = _connect(sock_path)
        res1 = c.generate(prompt="hello")
        assert res1["text"] == "reply-1"
        assert engine.unload_calls == 0
        assert engine.is_loaded is True
        c.close()

        # Wait past keep_alive=0.08s: daemon unloads weights while staying alive
        deadline = time.monotonic() + 2.0
        while engine.is_loaded and time.monotonic() < deadline:
            time.sleep(0.02)
        assert engine.is_loaded is False
        assert engine.unload_calls == 1

        # Status reports loaded=False without reloading weights
        c = _connect(sock_path)
        st = c.status()
        assert st["loaded"] is False
        assert st["keep_alive"] == 0.08
        assert engine.load_calls == 0

        # 2nd request: transparently reloads weights and succeeds
        res2 = c.generate(prompt="world")
        assert res2["text"] == "reply-2"
        assert engine.load_calls == 1
        assert engine.is_loaded is True

        c.shutdown()
        c.close()
    finally:
        daemon._close_listener()
        thread.join(timeout=3.0)


def test_keep_alive_zero_unloads_immediately_after_each_request(tmp_path):
    engine = _FakeEngine()
    daemon = ServingDaemon(
        "fake-model", "cpu-test", mode="triton", idle_timeout=5.0,
        engine=engine, keep_alive=0,
    )
    sock_path, thread = _start_daemon_on_unix_socket(tmp_path, daemon)
    try:
        c = _connect(sock_path)
        res1 = c.generate(prompt="first")
        assert res1["text"] == "reply-1"
        assert engine.unload_calls == 1
        assert engine.is_loaded is False

        res2 = c.generate(prompt="second")
        assert res2["text"] == "reply-2"
        assert engine.load_calls == 1
        assert engine.unload_calls == 2
        assert engine.is_loaded is False

        c.shutdown()
        c.close()
    finally:
        daemon._close_listener()
        thread.join(timeout=3.0)


def test_keep_alive_minus_one_stays_resident(tmp_path):
    engine = _FakeEngine()
    daemon = ServingDaemon(
        "fake-model", "cpu-test", mode="triton", idle_timeout=5.0,
        engine=engine, keep_alive=-1,
    )
    sock_path, thread = _start_daemon_on_unix_socket(tmp_path, daemon)
    try:
        c = _connect(sock_path)
        res = c.generate(prompt="stay")
        assert res["text"] == "reply-1"
        c.close()

        time.sleep(0.15)
        assert engine.unload_calls == 0
        assert engine.is_loaded is True

        c = _connect(sock_path)
        c.shutdown()
        c.close()
    finally:
        daemon._close_listener()
        thread.join(timeout=3.0)
