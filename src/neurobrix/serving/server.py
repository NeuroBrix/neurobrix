"""
ServingDaemon — Long-lived process that holds the InferenceEngine.

Listens on Unix domain socket for JSON-RPC requests.
Single-user, single-connection for V1.

Supports background daemonization: fork to background, parent waits
for ready signal, prints status, exits. Single terminal workflow:
    neurobrix serve → neurobrix chat → neurobrix stop

ZERO HARDCODE: Model name and hardware profile from CLI args.
"""

import os
import sys
import signal
import socket
import traceback
from typing import Any, Dict, Optional

from neurobrix.serving.engine import InferenceEngine
from neurobrix.serving.protocol import (
    SOCKET_PATH, PID_PATH, LOG_PATH, DAEMON_DIR,
    send_message, recv_message,
    make_response,
)


class ServingDaemon:
    """
    Long-running process that holds InferenceEngine with weights in VRAM.
    Accepts connections via Unix socket, dispatches to engine methods.
    """

    def __init__(
        self,
        model_name: str,
        hardware_id: str,
        mode: str = "compiled",
        idle_timeout: float = 1800.0,
    ):
        self._engine = InferenceEngine(model_name, hardware_id, mode)
        self._idle_timeout = idle_timeout
        self._running = False
        self._server_socket: Optional[socket.socket] = None
        self._ready_fd: Optional[int] = None  # Write end of pipe for ready signal

    def start(self, foreground: bool = False) -> None:
        """
        Load model and start serving.

        If foreground=True, runs in current process (blocks terminal).
        If foreground=False (default), forks to background and signals parent when ready.
        """
        if not foreground:
            self._daemonize()
            # After _daemonize(), we are in the child process
            # stdout/stderr redirected to LOG_PATH

        self._start_serving()

    def _daemonize(self) -> None:
        """
        Fork to background. Parent waits for ready signal, then exits.
        Child continues execution with stdout/stderr → LOG_PATH.

        Uses a pipe for clean ready signaling (no polling).
        """
        # Create pipe: parent reads, child writes "ready" when model loaded
        read_fd, write_fd = os.pipe()

        pid = os.fork()

        if pid > 0:
            # ── PARENT PROCESS ──
            os.close(write_fd)

            # Wait for child to signal ready (or die)
            print(f"[Serve] Loading model in background (PID {pid})...")
            print(f"[Serve] Log: {LOG_PATH}")

            read_pipe = os.fdopen(read_fd, 'r')
            try:
                # Block until child writes or pipe closes (child died)
                msg = read_pipe.read()
            finally:
                read_pipe.close()

            if msg.startswith("ready:"):
                # Child loaded successfully
                info = msg[6:].strip()
                print(f"[Serve] {info}")
                print(f"[Serve] Use 'neurobrix chat' to connect")
                print(f"[Serve] Use 'neurobrix stop' to shutdown")
                sys.exit(0)
            elif msg.startswith("error:"):
                print(f"[Serve] Failed: {msg[6:].strip()}")
                sys.exit(1)
            else:
                # Pipe closed without message → child crashed
                print(f"[Serve] Daemon process died unexpectedly. Check {LOG_PATH}")
                sys.exit(1)

        else:
            # ── CHILD PROCESS (daemon) ──
            os.close(read_fd)
            self._ready_fd = write_fd

            # Detach from terminal (new session)
            os.setsid()

            # Redirect stdout/stderr to log file
            DAEMON_DIR.mkdir(parents=True, exist_ok=True)
            log_fd = os.open(str(LOG_PATH), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            os.dup2(log_fd, sys.stdout.fileno())
            os.dup2(log_fd, sys.stderr.fileno())
            os.close(log_fd)

            # Redirect stdin from /dev/null
            devnull = os.open(os.devnull, os.O_RDONLY)
            os.dup2(devnull, sys.stdin.fileno())
            os.close(devnull)

    def _signal_ready(self, message: str) -> None:
        """Signal parent process that daemon is ready (or failed)."""
        if self._ready_fd is not None:
            try:
                os.write(self._ready_fd, message.encode())
                os.close(self._ready_fd)
            except OSError:
                pass
            self._ready_fd = None

    def _start_serving(self) -> None:
        """Load model, bind socket, enter serve loop."""
        # Signal handling
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        # Ensure daemon directory exists
        DAEMON_DIR.mkdir(parents=True, exist_ok=True)

        # Clean stale socket
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()

        # Write PID
        PID_PATH.write_text(str(os.getpid()))

        try:
            # Load model (weights to VRAM)
            self._engine.load()

            # Bind socket
            self._server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._server_socket.bind(str(SOCKET_PATH))
            self._server_socket.listen(1)
            self._server_socket.settimeout(self._idle_timeout)

            strategy = "warm" if self._engine._warm_serving else "cold"
            family = self._engine.family or "unknown"
            ready_msg = (
                f"ready:Model loaded — {self._engine.model_name} "
                f"({family}, {strategy} serving)"
            )
            self._signal_ready(ready_msg)

            print(f"\n[Daemon] Listening on {SOCKET_PATH}")
            print(f"[Daemon] Idle timeout: {self._idle_timeout}s")

            self._running = True
            self._serve_loop()

        except Exception as e:
            self._signal_ready(f"error:{e}")
            raise

        finally:
            self._cleanup()

    def _serve_loop(self) -> None:
        """Accept connections and handle requests."""
        while self._running:
            try:
                conn, _ = self._server_socket.accept()
            except socket.timeout:
                print(f"[Daemon] Idle timeout ({self._idle_timeout}s) — shutting down")
                break
            except OSError:
                if self._running:
                    print("[Daemon] Socket error")
                break

            try:
                self._handle_connection(conn)
            except Exception as e:
                print(f"[Daemon] Connection error: {e}")
                traceback.print_exc()
            finally:
                conn.close()

    def _handle_connection(self, conn: socket.socket) -> None:
        """Handle a single client connection (may have multiple requests)."""
        conn.settimeout(None)  # No timeout on established connections

        while self._running:
            request = recv_message(conn)
            if request is None:
                break  # Client disconnected

            response = self._dispatch(request)
            send_message(conn, response)

            # Check if shutdown was requested
            if request.get("method") == "shutdown":
                self._running = False
                break

    def _dispatch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch JSON-RPC request to engine method."""
        method = request.get("method", "")
        params = request.get("params", {})

        try:
            if method == "generate":
                output_path = params.pop("output_path", None)
                result = self._engine.generate(**params)
                # For non-LLM: save output file, return path instead of raw tensors
                if output_path and self._engine.family != "llm" and "outputs" in result:
                    saved = self._engine.save_output(result["outputs"], output_path)
                    result.pop("outputs", None)
                    result["output_path"] = saved
                elif "outputs" in result:
                    # Can't serialize raw tensors — drop them
                    result.pop("outputs", None)
                return make_response(result)

            elif method == "chat":
                message = params.get("message", "")
                kwargs = {k: v for k, v in params.items() if k != "message"}
                # Track summarization count before call
                prev_count = 0
                if self._engine._session is not None:
                    prev_count = self._engine._session._summarization_count
                text = self._engine.chat(message, **kwargs)
                context = {}
                summarized = False
                if self._engine._session is not None:
                    context = self._engine._session.get_context_info()
                    summarized = self._engine._session._summarization_count > prev_count
                return make_response({"text": text, "context": context, "summarized": summarized})

            elif method == "new_chat":
                self._engine.new_conversation()
                return make_response({"status": "ok"})

            elif method == "status":
                return make_response(self._engine.get_status())

            elif method == "shutdown":
                self._engine.unload()
                return make_response({"status": "ok"})

            else:
                return make_response(error=f"Unknown method: {method}")

        except Exception as e:
            traceback.print_exc()
            return make_response(error=str(e))

    def _handle_signal(self, signum, frame):
        """Graceful shutdown on SIGTERM/SIGINT."""
        sig_name = signal.Signals(signum).name
        print(f"\n[Daemon] Received {sig_name} — shutting down")
        self._running = False
        # Close server socket to unblock accept()
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except Exception:
                pass

    def _cleanup(self) -> None:
        """Clean up socket and PID files."""
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except Exception:
                pass

        if SOCKET_PATH.exists():
            try:
                SOCKET_PATH.unlink()
            except Exception:
                pass

        if PID_PATH.exists():
            try:
                PID_PATH.unlink()
            except Exception:
                pass

        print("[Daemon] Cleaned up")
