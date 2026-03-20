"""
DaemonClient — Connect to running ServingDaemon via IPC.

Provides static is_running() check and request/response helpers.
Used by CLI commands (chat, run warm-path) to communicate with daemon.

IPC transport:
  - Unix/macOS: AF_UNIX domain socket
  - Windows: AF_INET TCP on localhost:19384
"""

import os
import socket
from typing import Any, Dict, Optional

from neurobrix.serving.protocol import (
    SOCKET_PATH, PID_PATH,
    IPC_FAMILY, IPC_ADDRESS, IS_WINDOWS,
    send_message, recv_message,
    make_request,
)


class DaemonClient:
    """
    Client for communicating with ServingDaemon over IPC socket.

    Usage:
        if DaemonClient.is_running():
            client = DaemonClient()
            client.connect()
            result = client.generate(prompt="Hello")
            client.close()
    """

    def __init__(self):
        self._sock: Optional[socket.socket] = None

    @staticmethod
    def is_running() -> bool:
        """Check if daemon is alive: PID file exists and process running."""
        if not PID_PATH.exists():
            return False

        # Unix: also check socket file exists
        if not IS_WINDOWS and (SOCKET_PATH is None or not SOCKET_PATH.exists()):
            return False

        try:
            pid = int(PID_PATH.read_text().strip())
            if IS_WINDOWS:
                # Windows: use ctypes to check if process exists
                import ctypes
                kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
                PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
                if handle:
                    kernel32.CloseHandle(handle)
                    return True
                return False
            else:
                os.kill(pid, 0)  # Unix: signal 0 = check if process exists
                return True
        except (ValueError, ProcessLookupError, PermissionError, OSError):
            return False

    @staticmethod
    def get_pid() -> Optional[int]:
        """Get daemon PID, or None if not running."""
        if not PID_PATH.exists():
            return None
        try:
            return int(PID_PATH.read_text().strip())
        except (ValueError, OSError):
            return None

    def connect(self) -> None:
        """Connect to daemon socket."""
        if IS_WINDOWS:
            # TCP: try connecting to localhost port
            self._sock = socket.socket(IPC_FAMILY, socket.SOCK_STREAM)
            try:
                self._sock.connect(IPC_ADDRESS)
            except (ConnectionRefusedError, OSError):
                self._sock.close()
                self._sock = None
                raise RuntimeError(
                    "ZERO FALLBACK: Cannot connect to daemon on localhost:19384. "
                    "Start daemon first: neurobrix serve --model <name>"
                )
        else:
            # Unix domain socket
            if SOCKET_PATH is None or not SOCKET_PATH.exists():
                raise RuntimeError(
                    "ZERO FALLBACK: Daemon socket not found. "
                    "Start daemon first: neurobrix serve --model <name>"
                )
            self._sock = socket.socket(IPC_FAMILY, socket.SOCK_STREAM)
            try:
                self._sock.connect(str(SOCKET_PATH))
            except ConnectionRefusedError:
                self._sock.close()
                self._sock = None
                raise RuntimeError(
                    "ZERO FALLBACK: Daemon socket exists but connection refused. "
                    "Daemon may have crashed. Check with: neurobrix serve"
                )

    def close(self) -> None:
        """Close connection."""
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def send(self, method: str, **params) -> Dict[str, Any]:
        """Send JSON-RPC request and return response."""
        if self._sock is None:
            raise RuntimeError("ZERO FALLBACK: Not connected. Call connect() first.")

        request = make_request(method, **params)
        send_message(self._sock, request)

        response = recv_message(self._sock)
        if response is None:
            raise RuntimeError("ZERO FALLBACK: Daemon disconnected.")

        if "error" in response:
            raise RuntimeError(f"Daemon error: {response['error']}")

        return response.get("result", {})

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Send generate request."""
        return self.send("generate", prompt=prompt, **kwargs)

    def chat(self, message: str, **kwargs) -> Dict[str, Any]:
        """Send chat request. Returns {text, context}."""
        return self.send("chat", message=message, **kwargs)

    def new_chat(self) -> Dict[str, Any]:
        """Reset conversation history."""
        return self.send("new_chat")

    def status(self) -> Dict[str, Any]:
        """Get daemon status."""
        return self.send("status")

    def shutdown(self) -> Dict[str, Any]:
        """Request daemon shutdown."""
        return self.send("shutdown")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()
