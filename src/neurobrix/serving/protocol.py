"""
NeuroBrix Serving Protocol — Length-prefixed JSON-RPC over IPC.

Wire format: [4 bytes: uint32 big-endian message length][JSON payload]

IPC transport:
  - Unix/macOS: AF_UNIX domain socket (zero network overhead)
  - Windows: AF_INET TCP on localhost — a fixed port for the default instance,
    a port the OS hands out (recorded in `<stem>.port`) for an instance named
    by NBX_SOCKET_PATH, so instances never collide (Studio request 8)

ZERO HARDCODE: No HTTP, no REST, no gRPC.
Same-machine IPC only — minimal overhead for GPU-bound workloads.
"""

import sys
import json
import struct
import socket
from typing import Any, Dict, Optional
from pathlib import Path

# Platform detection
IS_WINDOWS = sys.platform == "win32"

# Daemon file locations
DAEMON_DIR = Path.home() / ".neurobrix"

#: The fixed loopback port of the DEFAULT Windows instance (no `NBX_SOCKET_PATH`):
#: the historical transport, kept byte-for-byte for the instance that never
#: named itself.
DEFAULT_WINDOWS_PORT = 19384


class Instance:
    """One daemon instance's identity: where it listens and which files carry
    its pid, log and (on Windows) its port. `NBX_SOCKET_PATH` names an
    instance on EVERY platform (Studio request 8): on Unix it is the socket
    path; on Windows there is no socket file, so the stem names the pid, log
    and port files and the daemon takes a port the OS hands out at bind
    (`bind_address()` port 0), records it in `<stem>.port`, and every client
    of that instance reads it there (`connect_address()`). Two instances
    never share a port; the default instance keeps the fixed port.

    Built once at import for this process (`_configure`), and again in a
    test for a platform that is not this one."""

    def __init__(self, platform: str, env: Dict[str, str]) -> None:
        self.is_windows = platform == "win32"
        named = env.get("NBX_SOCKET_PATH")
        stem = Path(named) if named else None
        if self.is_windows:
            self.socket_path = None
            self.family = socket.AF_INET
            if stem is not None:
                self.pid_path = stem.with_suffix(".pid")
                self.log_path = stem.with_suffix(".log")
                self.port_path = stem.with_suffix(".port")
                self._fixed_port = None
            else:
                self.pid_path = DAEMON_DIR / "daemon.pid"
                self.log_path = DAEMON_DIR / "daemon.log"
                self.port_path = None
                self._fixed_port = DEFAULT_WINDOWS_PORT
        else:
            # NBX_SOCKET_PATH: per-instance socket override so independent
            # daemons can coexist (harness finding 2026-08-13: parallel pinned
            # runners collided on the single default socket). Both the server
            # AND every client read the same env. Default (env absent) is
            # byte-for-byte the historical path; a named instance's pid file
            # (the "already running" guard) and log follow the socket stem so
            # two instances never adjudicate each other's liveness.
            self.socket_path = stem if stem is not None else DAEMON_DIR / "daemon.sock"
            self.family = socket.AF_UNIX
            self.pid_path = (self.socket_path.with_suffix(".pid") if stem is not None
                             else DAEMON_DIR / "daemon.pid")
            self.log_path = (self.socket_path.with_suffix(".log") if stem is not None
                             else DAEMON_DIR / "daemon.log")
            self.port_path = None
            self._fixed_port = None

    # -- addresses --------------------------------------------------------
    def bind_address(self):
        """What the daemon binds: the socket path on Unix; on Windows the
        fixed port for the default instance, port 0 (the OS chooses a free
        one) for a named instance."""
        if not self.is_windows:
            return str(self.socket_path)
        return ("127.0.0.1", self._fixed_port if self._fixed_port is not None else 0)

    def record_bound_port(self, sock: socket.socket) -> Optional[int]:
        """After `bind`, write the port the OS handed a named Windows instance
        to its port file, so its clients find it. Returns the port."""
        if not self.is_windows:
            return None
        port = int(sock.getsockname()[1])
        if self.port_path is not None:
            self.port_path.parent.mkdir(parents=True, exist_ok=True)
            self.port_path.write_text(str(port))
        return port

    def connect_address(self):
        """What a client connects to. A named Windows instance whose port file
        is absent has no daemon: refused by name, never a guess at a port."""
        if not self.is_windows:
            return str(self.socket_path)
        if self._fixed_port is not None:
            return ("127.0.0.1", self._fixed_port)
        if self.port_path is None or not self.port_path.exists():
            raise RuntimeError(
                f"ZERO FALLBACK: no daemon for instance {self.port_path.stem if self.port_path else '?'} "
                f"(no port file {self.port_path}). Start it first: neurobrix serve --model <name>")
        return ("127.0.0.1", int(self.port_path.read_text().strip()))

    def endpoint(self) -> Dict[str, Any]:
        """Where this instance's daemon listens, as a record (Studio request 1)."""
        if not self.is_windows:
            return {"kind": "unix", "path": str(self.socket_path)}
        if self._fixed_port is not None:
            return {"kind": "tcp", "address": "127.0.0.1", "port": self._fixed_port}
        port = (int(self.port_path.read_text().strip())
                if self.port_path is not None and self.port_path.exists() else None)
        return {"kind": "tcp", "address": "127.0.0.1", "port": port,
                "port_file": str(self.port_path)}

    def instance_files(self):
        """The files a stopped or crashed instance leaves behind, to remove."""
        return [p for p in (self.socket_path, self.pid_path, self.port_path) if p is not None]


def _configure(platform: str, env) -> Instance:
    return Instance(platform, dict(env))


import os as _os
INSTANCE = _configure(sys.platform, _os.environ)
SOCKET_PATH = INSTANCE.socket_path
PID_PATH = INSTANCE.pid_path
LOG_PATH = INSTANCE.log_path
PORT_PATH = INSTANCE.port_path
IPC_FAMILY = INSTANCE.family
#: The bind address of this process's instance. On a named Windows instance
#: the port is 0 until the daemon binds; clients use `INSTANCE.connect_address()`.
IPC_ADDRESS = INSTANCE.bind_address()

# Protocol constants
HEADER_SIZE = 4  # uint32 big-endian
MAX_MESSAGE_SIZE = 64 * 1024 * 1024  # 64MB safety limit


def send_message(sock: socket.socket, data: Dict[str, Any]) -> None:
    """Send a length-prefixed JSON message over a socket."""
    payload = json.dumps(data, default=str).encode("utf-8")
    header = struct.pack(">I", len(payload))
    sock.sendall(header + payload)


def recv_message(sock: socket.socket) -> Optional[Dict[str, Any]]:
    """Receive a length-prefixed JSON message from a socket."""
    # Read header
    header = _recv_exact(sock, HEADER_SIZE)
    if header is None:
        return None

    msg_len = struct.unpack(">I", header)[0]
    if msg_len > MAX_MESSAGE_SIZE:
        raise RuntimeError(f"Message too large: {msg_len} bytes (max {MAX_MESSAGE_SIZE})")

    # Read payload
    payload = _recv_exact(sock, msg_len)
    if payload is None:
        return None

    return json.loads(payload.decode("utf-8"))


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    """Receive exactly n bytes from socket, or None on disconnect."""
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


#: The wire protocol's version. It changes when a method's request or response
#: shape changes; adding a method does not change it. Every response envelope
#: carries it beside the engine's version so a client refuses what it does not
#: know instead of guessing (Studio requests 1 and 7). One number, read here.
PROTOCOL_VERSION = 1


def endpoint() -> Dict[str, Any]:
    """Where this process's daemon instance listens, as a record (Studio request 1)."""
    return INSTANCE.endpoint()


def make_request(method: str, **params) -> Dict[str, Any]:
    """Build a JSON-RPC request."""
    return {
        "method": method,
        "params": params,
    }


def make_response(result: Any = None, error: Optional[str] = None) -> Dict[str, Any]:
    """Build a JSON-RPC response. Every envelope names the protocol and the
    engine that produced it."""
    from neurobrix import __version__
    head = {"protocol": PROTOCOL_VERSION, "engine": __version__}
    if error is not None:
        return {**head, "error": error}
    return {**head, "result": result}
