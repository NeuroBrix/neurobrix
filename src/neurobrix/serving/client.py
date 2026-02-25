"""
DaemonClient — Connect to running ServingDaemon via Unix socket.

Provides static is_running() check and request/response helpers.
Used by CLI commands (chat, run warm-path) to communicate with daemon.
"""

import os
import socket
from typing import Any, Dict, Optional

from neurobrix.serving.protocol import (
    SOCKET_PATH, PID_PATH,
    send_message, recv_message,
    make_request,
)


class DaemonClient:
    """
    Client for communicating with ServingDaemon over Unix socket.

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
        """Check if daemon is alive: PID file exists, process running, socket exists."""
        if not PID_PATH.exists() or not SOCKET_PATH.exists():
            return False

        try:
            pid = int(PID_PATH.read_text().strip())
            os.kill(pid, 0)  # Signal 0 = check if process exists
            return True
        except (ValueError, ProcessLookupError, PermissionError):
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
        if not SOCKET_PATH.exists():
            raise RuntimeError(
                "ZERO FALLBACK: Daemon socket not found. "
                "Start daemon first: neurobrix serve --model <name> --hardware <profile>"
            )

        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
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
