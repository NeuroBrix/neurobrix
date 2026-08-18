"""
NeuroBrix Serving Protocol — Length-prefixed JSON-RPC over IPC.

Wire format: [4 bytes: uint32 big-endian message length][JSON payload]

IPC transport:
  - Unix/macOS: AF_UNIX domain socket (zero network overhead)
  - Windows: AF_INET TCP on localhost:19384 (cross-platform)

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
PID_PATH = DAEMON_DIR / "daemon.pid"
LOG_PATH = DAEMON_DIR / "daemon.log"

# IPC transport — platform-adaptive
if IS_WINDOWS:
    IPC_PORT = 19384
    IPC_ADDRESS = ("127.0.0.1", IPC_PORT)
    IPC_FAMILY = socket.AF_INET
    SOCKET_PATH = None  # No Unix socket on Windows
    import os as _os_w
    if _os_w.environ.get("NBX_SOCKET_PATH"):
        import sys as _sys_w
        print("[NeuroBrix] WARNING: NBX_SOCKET_PATH is ignored on "
              "Windows (fixed TCP port transport) — per-instance "
              "isolation is NOT in effect.", file=_sys_w.stderr)
else:
    # NBX_SOCKET_PATH: per-instance socket override so independent
    # daemons can coexist (harness finding 2026-08-13: parallel pinned
    # runners collided on the single default socket — one row's daemon
    # held it while another row's runner bound it, "[Errno 98] Address
    # already in use"; the GPU guard watches processes, not sockets).
    # Both the server AND every client read the same env, so a runner
    # that sets it gets a fully isolated daemon channel. Default
    # (env absent) is byte-for-byte the historical path.
    import os as _os
    _sock_env = _os.environ.get("NBX_SOCKET_PATH")
    SOCKET_PATH = (Path(_sock_env) if _sock_env
                   else DAEMON_DIR / "daemon.sock")
    if _sock_env:
        # A per-instance channel isolates the WHOLE identity: pid file
        # (the "already running" guard) and log follow the socket stem
        # so two instances never adjudicate each other's liveness or
        # interleave logs.
        PID_PATH = SOCKET_PATH.with_suffix(".pid")
        LOG_PATH = SOCKET_PATH.with_suffix(".log")
    IPC_ADDRESS = str(SOCKET_PATH)
    IPC_FAMILY = socket.AF_UNIX

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


def make_request(method: str, **params) -> Dict[str, Any]:
    """Build a JSON-RPC request."""
    return {
        "method": method,
        "params": params,
    }


def make_response(result: Any = None, error: Optional[str] = None) -> Dict[str, Any]:
    """Build a JSON-RPC response."""
    if error is not None:
        return {"error": error}
    return {"result": result}
