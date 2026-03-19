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
else:
    SOCKET_PATH = DAEMON_DIR / "daemon.sock"
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
