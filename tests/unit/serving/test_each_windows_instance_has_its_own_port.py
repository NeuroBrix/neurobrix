"""On Windows there is no socket file, so `NBX_SOCKET_PATH` used to isolate
nothing: every instance bound the one fixed loopback port (Studio request 8).
Now the env names an instance on every platform: on Windows a named instance
binds port 0, records the port the OS handed it in `<stem>.port`, its clients
read it there, and `endpoint()` reports it; the default instance keeps the
fixed port. Tested here by configuring the protocol for `win32` on this
machine — a loopback TCP bind is the same call on both.

Injection: with `record_bound_port` writing nothing, the first test failed at
the port file; with `connect_address` returning the fixed port for a named
instance, the two instances read the same port.
"""
from __future__ import annotations

import socket

import pytest

from neurobrix.serving.protocol import DEFAULT_WINDOWS_PORT, _configure


def _bind(inst):
    s = socket.socket(inst.family, socket.SOCK_STREAM)
    s.bind(inst.bind_address())
    inst.record_bound_port(s)
    return s


def test_two_named_windows_instances_take_two_ports_and_each_client_finds_its_own(tmp_path):
    a = _configure("win32", {"NBX_SOCKET_PATH": str(tmp_path / "a.sock")})
    b = _configure("win32", {"NBX_SOCKET_PATH": str(tmp_path / "b.sock")})
    assert a.bind_address() == ("127.0.0.1", 0) and a.port_path == tmp_path / "a.port"
    sa, sb = _bind(a), _bind(b)
    try:
        pa, pb = sa.getsockname()[1], sb.getsockname()[1]
        assert pa != pb
        assert a.connect_address() == ("127.0.0.1", pa)
        assert b.connect_address() == ("127.0.0.1", pb)
        assert a.endpoint() == {"kind": "tcp", "address": "127.0.0.1", "port": pa,
                                "port_file": str(tmp_path / "a.port")}
        assert a.pid_path == tmp_path / "a.pid" and a.log_path == tmp_path / "a.log"
        assert set(a.instance_files()) == {a.pid_path, a.port_path}
    finally:
        sa.close(); sb.close()


def test_the_default_windows_instance_keeps_the_fixed_port():
    d = _configure("win32", {})
    assert d.bind_address() == ("127.0.0.1", DEFAULT_WINDOWS_PORT)
    assert d.connect_address() == ("127.0.0.1", DEFAULT_WINDOWS_PORT)
    assert d.port_path is None
    assert d.endpoint() == {"kind": "tcp", "address": "127.0.0.1", "port": DEFAULT_WINDOWS_PORT}


def test_a_named_instance_without_a_daemon_is_refused_by_name(tmp_path):
    a = _configure("win32", {"NBX_SOCKET_PATH": str(tmp_path / "a.sock")})
    with pytest.raises(RuntimeError, match="no daemon for instance a"):
        a.connect_address()


def test_unix_instances_are_unchanged(tmp_path):
    u = _configure("linux", {"NBX_SOCKET_PATH": str(tmp_path / "u.sock")})
    assert u.bind_address() == str(tmp_path / "u.sock") == u.connect_address()
    assert u.endpoint() == {"kind": "unix", "path": str(tmp_path / "u.sock")}
    assert u.pid_path == tmp_path / "u.pid" and u.port_path is None
    d = _configure("linux", {})
    assert d.socket_path.name == "daemon.sock" and d.pid_path.name == "daemon.pid"
