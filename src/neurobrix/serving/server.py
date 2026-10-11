"""
ServingDaemon — Long-lived process that holds the InferenceEngine.

Listens on IPC socket for length-prefixed JSON requests (serving/protocol.py).

THE DISPATCHER (Studio requests 3-4; doctrine: docs/reference/studio-engine-requests.md,
"the dispatcher's shape", dfe596c5):

  * One thread per connection; the accept loop only accepts. A connection that
    is running a work request is the WORK channel of that request; any other
    connection is a CONTROL channel and is answered while the work runs.
  * One inference at a time: the work methods (`generate`, `chat`, `complete`)
    and the session mutation `new_chat` take ONE inference lock. `status`,
    `cancel`, `template` and `shutdown` never wait on it while a generation
    runs (`shutdown` takes it only after the running work has stopped, to
    unload). `new_chat` takes it although the 2026-09-16 shape listed it as
    control: it clears the conversation a running `chat` reads and appends to.
  * Every work request is an operation with an id (`op_id`, the client's or
    the daemon's), echoed by its reply, its errors and its stream events. The
    operations live in a table under its OWN lock (serving/operations.py).
  * `cancel {op_id}` sets the operation's stop flag, a threading.Event the work
    thread binds through `neurobrix.decode_stop`; both engines' decode loops
    read it once per token (stdlib-only seam, R33). The cancel is answered only
    after the decode loop returned AND every device of the plan was synced
    (`InferenceEngine.sync_devices`) — never on a timeout or a disconnect.

THE ORPHANED GENERATION (decided here, by the doctrine "no decode into the
void", decode_progress.py): when the client of a running work request vanishes
— its stream send fails, or the connection watcher sees it closed — the daemon
STOPS the generation at the next per-token seam, through the same stop flag,
and records the operation `orphaned`. It is never reported as a confirmed
cancel: a later `cancel` of it answers `already_finished` with state
`orphaned`; a client timeout is the client's own business and changes nothing
here. The inference lock is then free for the next client instead of being
held by a decode nobody reads.

Supports background daemonization:
  - Unix/macOS: fork to background, parent waits for ready signal
  - Windows: subprocess.Popen with CREATE_NEW_PROCESS_GROUP

Single terminal workflow:
    neurobrix serve → neurobrix chat → neurobrix stop

ZERO HARDCODE: Model name and hardware profile from CLI args; the dispatcher's
limits and timeouts from config/system.yml `serving:`.
"""

import json
import os
import sys
import select
import signal
import socket
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from neurobrix import decode_progress, decode_stop
from neurobrix.serving.engine import InferenceEngine
from neurobrix.serving.operations import OperationTable, Operation, serving_settings
from neurobrix.serving.protocol import (
    SOCKET_PATH, PID_PATH, LOG_PATH, DAEMON_DIR,
    IPC_FAMILY, IPC_ADDRESS, IS_WINDOWS, INSTANCE,
    send_message, recv_message, MalformedFrame, version_refusal,
    make_response,
)

#: The prefix of the one stdout line that carries the daemon's ready record.
READY_LINE = "NBX-READY"

#: The methods that run the model: each is an operation under the inference lock.
WORK_METHODS = ("generate", "chat", "complete")


def stream_listener(conn: socket.socket, engine, op_id: Optional[str] = None,
                    on_send_failure=None) -> "decode_progress.TokenListener":
    """The per-token listener of one streaming request. Every event carries
    `step`, `n`, `token`, `done` (and `op_id` when the request is an
    operation); for a family whose answer is text, it also carries the decoded
    delta (`text`, `rewind`) from the engine's own tokenizer (Studio request 2;
    `serving/text_stream.py`). Both engines emit at the same site, so one
    listener serves both (R30).

    `on_send_failure(exc)`, when given, is called instead of raising when the
    client is gone: the dispatcher stops the generation at the next seam (an
    orphan, see the module docstring) and no further event is sent."""
    decoder = None
    if _answers_in_text(engine):
        tokenizer = engine.tokenizer
        if tokenizer is not None:
            from neurobrix.serving.text_stream import TextDeltaDecoder
            decoder = TextDeltaDecoder(tokenizer)
    gone = []

    def _on_token(step: int, n: int, token_id: int, done: bool) -> None:
        if gone:
            return
        event = {"step": step, "n": n, "token": token_id, "done": done}
        if op_id is not None:
            event["op_id"] = op_id
        if decoder is not None:
            event.update(decoder.push(token_id))
        try:
            send_message(conn, {"stream": event})
        except OSError as e:
            if on_send_failure is None:
                raise
            gone.append(e)
            on_send_failure(e)

    return _on_token
def _answers_in_text(engine) -> bool:
    """Whether the loaded family's answer is text — read from the family's
    own declaration (output dispatch), never from a name."""
    from neurobrix.core.runtime.output_dispatch import get_output_format
    try:
        return get_output_format(engine.family or "") in ("txt", "mode_dependent")
    except RuntimeError:
        return False


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
        engine=None,
    ):
        # `engine` is injected only by a test (a fake engine that yields tokens
        # without a model); the daemon builds its InferenceEngine.
        self._engine = engine if engine is not None else InferenceEngine(model_name, hardware_id, mode)
        self._idle_timeout = idle_timeout
        self._running = False
        self._server_socket: Optional[socket.socket] = None
        # Where the launcher waits for the ready record, besides stdout: the
        # write end of its pipe (POSIX) or a file (a launcher with no pipe).
        self._ready_fd: Optional[int] = None
        self._ready_file = None
        self._settings = serving_settings()
        self._ops = OperationTable(self._settings["operations_retained"])
        self._inference_lock = threading.Lock()
        # Open connections and their threads, under their own lock.
        self._conn_lock = threading.Lock()
        self._connections: Dict[socket.socket, threading.Thread] = {}

    def start(self, foreground: bool = False, ready_fd: Optional[int] = None,
              ready_file=None) -> None:
        """
        Load model and start serving.

        If foreground=True, runs in current process (blocks terminal).
        If foreground=False (default):
          - Linux: fork to background, the parent waits on a pipe
          - macOS: a subprocess (fork+setsid breaks Metal), the parent waits on a pipe
          - Windows: a subprocess with CREATE_NEW_PROCESS_GROUP, the parent
            watches the instance's ready file

        `ready_fd` / `ready_file` are where the process that launched this
        one waits for the ready record (`--ready-fd` / `--ready-file`);
        stdout carries it in every case (`READY_LINE`).
        """
        self._ready_fd = ready_fd
        self._ready_file = ready_file
        if not foreground:
            if IS_WINDOWS or sys.platform == "darwin":
                # Windows + macOS: spawn subprocess (fork+setsid breaks Metal GPU on macOS)
                self._daemonize_subprocess()
                return  # Parent returns after spawning child
            else:
                self._daemonize_unix()
                # After _daemonize_unix(), we are in the child process
                # stdout/stderr redirected to LOG_PATH

        self._start_serving()

    def _daemonize_subprocess(self) -> None:
        """
        Spawn background daemon via subprocess (Windows + macOS).

        On macOS, os.fork() + os.setsid() breaks Metal GPU access
        (MTLCompilerService is per-session). Subprocess preserves the GPU context.
        On Windows, subprocess is the only option (no os.fork).

        The child is this command again in the foreground, told where its
        launcher waits: an inherited pipe on macOS (`pass_fds`), the
        instance's ready file on Windows, where subprocess passes no fd.
        """
        import subprocess

        DAEMON_DIR.mkdir(parents=True, exist_ok=True)

        # Build the command to re-execute ourselves in foreground mode
        cmd = [
            sys.executable, "-m", "neurobrix", "serve",
            "--model", self._engine.model_name,
            "--foreground",
        ]

        log_file = open(str(LOG_PATH), "w")

        popen_kwargs = {
            "stdout": log_file,
            "stderr": log_file,
            "stdin": subprocess.DEVNULL,
        }

        read_fd = None
        if IS_WINDOWS:
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            DETACHED_PROCESS = 0x00000008
            popen_kwargs["creationflags"] = CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS
            if INSTANCE.ready_path.exists():
                INSTANCE.ready_path.unlink()   # a dead daemon's record is not this one's
            cmd += ["--ready-file", str(INSTANCE.ready_path)]
        else:
            # macOS/Linux: start_new_session detaches without breaking GPU
            popen_kwargs["start_new_session"] = True
            read_fd, write_fd = os.pipe()
            popen_kwargs["pass_fds"] = (write_fd,)
            cmd += ["--ready-fd", str(write_fd)]

        proc = subprocess.Popen(cmd, **popen_kwargs)
        if read_fd is not None:
            os.close(write_fd)   # only the child holds it: its exit closes the pipe

        print(f"[Serve] Loading model in background (PID {proc.pid})...")
        print(f"[Serve] Log: {LOG_PATH}")

        record = wait_ready(proc, read_fd=read_fd,
                            ready_file=None if read_fd is not None else INSTANCE.ready_path,
                            poll_s=self._settings["ready_poll_s"])
        if not _announce(record):
            sys.exit(1)

    def _daemonize_unix(self) -> None:
        """
        Unix/macOS: fork to background. Parent waits for ready signal, then exits.
        Child continues execution with stdout/stderr → LOG_PATH.

        Uses a pipe for clean ready signaling (no polling).
        """
        # Create pipe: parent reads, child writes the ready record when model loaded
        read_fd, write_fd = os.pipe()

        pid = os.fork()

        if pid > 0:
            # ── PARENT PROCESS ──
            os.close(write_fd)

            # Wait for child to signal ready (or die)
            print(f"[Serve] Loading model in background (PID {pid})...")
            print(f"[Serve] Log: {LOG_PATH}")

            record = wait_ready(_ForkedChild(pid), read_fd=read_fd)
            sys.exit(0 if _announce(record) else 1)

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

    def _ready_record(self, error: Optional[BaseException] = None) -> Dict[str, Any]:
        """What the launcher learns at start-up, as one record: ready (bound,
        listening, model loaded) with where and what is served, or failed
        with why. The identity part is the one `neurobrix info --json` prints."""
        from neurobrix.serving.engine import daemon_identity
        rec = {"state": "failed" if error is not None else "ready", "pid": os.getpid(),
               "model": self._engine.model_name, **daemon_identity()}
        if error is not None:
            rec["error"] = f"{type(error).__name__}: {error}"
        else:
            rec["family"] = self._engine.family or "unknown"
            rec["serving"] = "warm" if self._engine._warm_serving else "cold"
        return rec

    def _signal_ready(self, record: Dict[str, Any]) -> None:
        """Say the ready record: one stdout line in every run (a launcher that
        reads the child's stdout, e.g. Studio), then the launcher's pipe or
        file when it gave one. A record that cannot be delivered is said in
        the log; the daemon serves on (its clients do not need the launcher)."""
        line = json.dumps(record, separators=(",", ":"))
        print(f"{READY_LINE} {line}", flush=True)
        if self._ready_fd is not None:
            fd, self._ready_fd = self._ready_fd, None
            try:
                os.write(fd, (line + "\n").encode())
            except OSError as e:
                print(f"[Daemon] ready signal not delivered on fd {fd}: {e}", flush=True)
            finally:
                os.close(fd)
        if self._ready_file is not None:
            path, self._ready_file = Path(self._ready_file), None
            tmp = path.with_name(path.name + ".tmp")
            try:
                tmp.write_text(line + "\n")
                os.replace(tmp, path)   # the launcher never reads half a record
            except OSError as e:
                print(f"[Daemon] ready signal not delivered to {path}: {e}", flush=True)

    def _start_serving(self) -> None:
        """Load model, bind socket, enter serve loop."""
        # Signal handling (SIGINT works on all platforms, SIGTERM on Unix only)
        signal.signal(signal.SIGINT, self._handle_signal)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self._handle_signal)

        # Ensure daemon directory exists
        DAEMON_DIR.mkdir(parents=True, exist_ok=True)

        # Clean stale socket (Unix only — TCP has no file)
        if SOCKET_PATH is not None and SOCKET_PATH.exists():
            SOCKET_PATH.unlink()

        # Write PID
        PID_PATH.write_text(str(os.getpid()))

        try:
            # Load model (weights to VRAM)
            self._engine.load()

            # Bind socket — platform-adaptive
            server_socket = socket.socket(IPC_FAMILY, socket.SOCK_STREAM)
            if IS_WINDOWS:
                # Allow port reuse on Windows (prevents "address already in use" after crash)
                server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(INSTANCE.bind_address())
            # A named Windows instance took the port the OS handed out: record
            # it where its clients read it (Studio request 8).
            INSTANCE.record_bound_port(server_socket)
            server_socket.listen(self._settings["listen_backlog"])

            self._signal_ready(self._ready_record())

            print(f"\n[Daemon] Listening on {INSTANCE.endpoint()}")
            print(f"[Daemon] Idle timeout: {self._idle_timeout}s")

            self.serve(server_socket)

        except Exception as e:
            self._signal_ready(self._ready_record(e))
            raise

        finally:
            self._cleanup()

    # ------------------------------------------------------------------
    # The accept loop: it only accepts, every connection gets its thread.
    # ------------------------------------------------------------------
    def serve(self, server_socket: socket.socket) -> None:
        """Serve on a bound, listening socket until shutdown, a signal or the
        idle timeout. The daemon calls it after loading; a test calls it on a
        socket of its own with a fake engine."""
        self._server_socket = server_socket
        self._server_socket.settimeout(self._idle_timeout)
        self._running = True
        self._serve_loop()

    def _serve_loop(self) -> None:
        """Accept connections; hand each to its own thread."""
        while self._running:
            try:
                conn, _ = self._server_socket.accept()
            except socket.timeout:
                if self._open_connections():
                    continue   # a client is connected (maybe generating): not idle
                print(f"[Daemon] Idle timeout ({self._idle_timeout}s) — shutting down")
                break
            except OSError:
                if self._running:
                    print("[Daemon] Socket error")
                break
            if not self._running:
                conn.close()
                break
            thread = threading.Thread(target=self._serve_connection, args=(conn,),
                                      name="nbx-conn", daemon=True)
            with self._conn_lock:
                self._connections[conn] = thread
            thread.start()
        self._running = False
        # The loop owns its listener and closes it only once it has left
        # accept(): closed from another thread, a macOS listener drops the
        # queued wake connection and leaves accept() blocked for good.
        try:
            self._server_socket.close()
        except OSError as e:
            print(f"[Daemon] Listener close: {e}")
        self._drain()

    def _open_connections(self) -> int:
        with self._conn_lock:
            return len(self._connections)

    def _drain(self) -> None:
        """At exit: stop what still runs, let the connection threads deliver
        the replies of the work they stopped, then close every connection."""
        self._ops.stop_all("shutdown")
        deadline = time.monotonic() + self._settings["drain_timeout_s"]
        if not self._inference_lock.acquire(timeout=self._settings["drain_timeout_s"]):
            print("[Daemon] A generation did not stop within drain_timeout_s — exiting beside it")
        else:
            self._inference_lock.release()
        with self._conn_lock:
            items = list(self._connections.items())
        for conn, _ in items:
            try:
                conn.shutdown(socket.SHUT_RD)   # unblocks a thread waiting for its next request
            except OSError as e:
                print(f"[Daemon] Connection already closed at drain: {e}")
        for _, thread in items:
            thread.join(max(0.0, deadline - time.monotonic()))

    # ------------------------------------------------------------------
    # One connection.
    # ------------------------------------------------------------------
    def _serve_connection(self, conn: socket.socket) -> None:
        try:
            self._handle_connection(conn)
        except OSError as e:
            # The client went away while we wrote to or read from it.
            print(f"[Daemon] Connection lost: {e}")
        except Exception as e:
            print(f"[Daemon] Connection error: {e}")
            traceback.print_exc()
        finally:
            with self._conn_lock:
                self._connections.pop(conn, None)
            conn.close()

    def _handle_connection(self, conn: socket.socket) -> None:
        """Serve one client connection (may carry multiple requests, in order)."""
        conn.settimeout(None)  # No timeout on established connections
        from neurobrix.core.runtime.phase_trace import mark as _phase_mark

        while self._running:
            try:
                request = recv_message(conn)
            except MalformedFrame as e:
                send_message(conn, make_response(error=f"malformed frame: {e}"))
                if not e.resync:
                    return   # no frame boundary left to trust: close
                continue
            if request is None:
                break  # Client disconnected

            refused = version_refusal(request)
            if refused is not None:
                send_message(conn, make_response(error=refused))
                return   # a client of another protocol: told why, then closed

            method = request.get("method", "")
            _phase_mark(f"server.recv.{method or '?'}")
            response = self._dispatch(request, conn)
            send_message(conn, response)
            _phase_mark("server.sent")

            if method == "shutdown" and "result" in response:
                self._close_listener()
                break

    # ------------------------------------------------------------------
    # Dispatch.
    # ------------------------------------------------------------------
    def _dispatch(self, request: Dict[str, Any], conn: socket.socket) -> Dict[str, Any]:
        """Dispatch one request. Work methods become operations under the
        inference lock; control methods never wait on a running generation.

        `conn` is the client connection — used by work requests to push
        per-token stream events ahead of the final response on the same
        length-prefixed wire, and to notice a client that vanished."""
        method = request.get("method", "")
        params = request.get("params", {})
        if params is None:
            params = {}
        if not isinstance(method, str) or not isinstance(params, dict):
            return make_response(
                error=f"malformed request: `method` must be a string and `params` an object, "
                      f"got {type(method).__name__} and {type(params).__name__}")

        if method in WORK_METHODS:
            return self._run_work(method, params, request.get("op_id"), conn)

        try:
            if method == "status":
                return make_response({**self._engine.get_status(),
                                      "operations": self._ops.live()})

            elif method == "cancel":
                return self._cancel(params)

            elif method == "template":
                return make_response({"text": self._engine.chat_template_text()})

            elif method == "new_chat":
                with self._inference_lock:
                    self._engine.new_conversation()
                return make_response({"status": "ok"})

            elif method == "shutdown":
                return self._shutdown()

            else:
                return make_response(error=f"Unknown method: {method}")

        except Exception as e:
            traceback.print_exc()
            return make_response(error=str(e))

    # -- work -------------------------------------------------------------
    def _run_work(self, method: str, params: Dict[str, Any], requested_id,
                  conn: socket.socket) -> Dict[str, Any]:
        try:
            op = self._ops.open(method, requested_id)
        except ValueError as e:
            return make_response(error=f"refused: {e}")

        stream = method == "generate" and bool(params.pop("stream", False))
        watcher = threading.Thread(target=self._watch_client, args=(conn, op),
                                   name=f"nbx-watch-{op.op_id}", daemon=True)
        watcher.start()

        with self._inference_lock:
            if not self._ops.start(op):
                # Stopped while it waited for the lock: no native work began.
                return self._work_reply(op, None, None)
            listener = self._step_listener(conn, op, stream)
            decode_progress.set_listener(listener)
            decode_stop.bind(op.stop_event)
            result, error = None, None
            try:
                result = self._call_engine(method, params)
            except Exception as e:
                traceback.print_exc()
                error = str(e)
            finally:
                honoured = decode_stop.honoured_at()
                decode_stop.unbind()
                decode_progress.clear_listener()
            # The devices are synced BEFORE the operation is finished: a cancel
            # waiting on `op.done` is acknowledged only after this returns.
            try:
                self._engine.sync_devices()
            except Exception as e:
                traceback.print_exc()
                error = f"{error}; device sync failed: {e}" if error else f"device sync failed: {e}"
                honoured = None   # an unsynced device is never an acknowledged stop
            state = self._ops.finish(op, honoured, error)
        return self._work_reply(op, state, result)

    def _step_listener(self, conn, op: Operation, stream: bool):
        """Counts the operation's steps for `status` and `cancel`; streams the
        event to the client when the request asked for it."""
        sender = (stream_listener(conn, self._engine, op.op_id,
                                  on_send_failure=lambda e: self._orphan(op, e))
                  if stream else None)

        def _on_token(step: int, n: int, token_id: int, done: bool) -> None:
            self._ops.step(op, n)
            if sender is not None:
                sender(step, n, token_id, done)

        return _on_token

    def _watch_client(self, conn: socket.socket, op: Operation) -> None:
        """While `op` runs, notice a client that vanished (its connection
        closed or reset) and orphan the operation. Data from a live client
        (a pipelined next request) ends the watch: the client is there."""
        poll = self._settings["orphan_poll_s"]
        while not op.done.is_set():
            try:
                readable, _, _ = select.select([conn], [], [], poll)
                if not readable:
                    continue
                if conn.recv(1, socket.MSG_PEEK) == b"":
                    self._orphan(op, "connection closed")
                return
            except (OSError, ValueError) as e:
                # ValueError: the socket was closed under us (fd -1).
                self._orphan(op, e)
                return

    def _orphan(self, op: Operation, why) -> None:
        _, prior = self._ops.request_stop(op.op_id, "orphan")
        if prior is None:
            print(f"[Daemon] Client of {op.op_id} vanished ({why}) — stopping it at the next token")

    def _call_engine(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if method == "generate":
            output_path = params.pop("output_path", None)
            result = self._engine.generate(**params)
            # For binary-output families (image, video, tts wav): save the
            # tensor to disk and return the path instead of raw tensors.
            # Text families (llm, vlm, multimodal-text, stt, audio_llm) put
            # the answer directly in the JSON response.
            from neurobrix.core.runtime.output_dispatch import get_output_format
            try:
                fmt = get_output_format(self._engine.family or "")
            except RuntimeError:
                fmt = "txt"
            if output_path and fmt != "txt" and "outputs" in result:
                # The request mode rides to save_output — multimodal-
                # strict families resolve their extension from it.
                saved = self._engine.save_output(
                    result["outputs"], output_path,
                    mode=params.get("mode"),
                    orig_hw=result.pop("_upscale_orig_hw", None))
                result.pop("outputs", None)
                result["output_path"] = saved
            elif "outputs" in result:
                # Can't serialize raw tensors — drop them
                result.pop("outputs", None)
            return result

        if method == "chat":
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
            return {"text": text, "context": context, "summarized": summarized}

        # complete
        messages = params.get("messages") or []
        tools = params.get("tools")
        kwargs = {k: v for k, v in params.items() if k not in ("messages", "tools")}
        return {"text": self._engine.complete(messages, tools=tools, **kwargs)}

    def _work_reply(self, op: Operation, state: Optional[str], result) -> Dict[str, Any]:
        snap = self._ops.snapshot(op)
        if snap["state"] == "failed":
            return make_response(error=snap.get("detail") or "failed", op_id=op.op_id)
        out = dict(result or {})
        out["op_id"] = op.op_id
        out["status"] = snap["state"]
        if snap["state"] != "completed":
            # cancelled / orphaned / stopped: what was produced before the stop
            out["steps"] = snap["steps"]
            if snap.get("detail"):
                out["detail"] = snap["detail"]
        return make_response(out, op_id=op.op_id)

    # -- control ----------------------------------------------------------
    def _cancel(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """`cancel {op_id}`. Every outcome is named — none is a silent ok:

          cancelled          this request stopped it; its decode loop returned
                             and the devices were synced before this reply
          already_cancelled  an earlier cancel stopped (or is stopping) it
          already_finished   it ended on its own, or was orphaned/stopped/failed;
                             `state` says which
          unknown_op         no operation has (or still remembers) this id
          cancel_pending     the stop is set but the loop did not return within
                             serving.cancel_ack_timeout_s — NOT a confirmation
        """
        op_id = params.get("op_id")
        if not isinstance(op_id, str) or not op_id:
            return make_response(error=f"cancel needs params.op_id (a non-empty string), got {op_id!r}")
        op, prior = self._ops.request_stop(op_id, "cancel")
        if prior == "unknown":
            return make_response({"outcome": "unknown_op", "op_id": op_id})
        if prior != "finished":
            if not op.done.wait(self._settings["cancel_ack_timeout_s"]):
                snap = self._ops.snapshot(op)
                return make_response({"outcome": "cancel_pending", "op_id": op_id,
                                      "state": snap["state"], "steps": snap["steps"]})
        snap = self._ops.snapshot(op)
        if snap["state"] == "cancelled":
            outcome = "cancelled" if prior is None else "already_cancelled"
        else:
            outcome = "already_finished"
        return make_response({"outcome": outcome, "op_id": op_id,
                              "state": snap["state"], "steps": snap["steps"]})

    def _shutdown(self) -> Dict[str, Any]:
        """Stop every running operation (`stopped`), wait for it to return,
        then unload under the inference lock. The listener closes once this
        reply is sent."""
        self._running = False
        for op in self._ops.stop_all("shutdown"):
            op.done.wait(self._settings["cancel_ack_timeout_s"])
        if not self._inference_lock.acquire(timeout=self._settings["cancel_ack_timeout_s"]):
            self._close_listener()
            return make_response(error="shutdown: the running generation did not stop within "
                                       "serving.cancel_ack_timeout_s; the daemon exits without unloading")
        try:
            self._engine.unload()
        finally:
            self._inference_lock.release()
        return make_response({"status": "ok"})

    def _close_listener(self) -> None:
        """Stop the accept loop from any thread. Closing a listening socket
        does NOT wake an `accept()` blocked in another thread (Linux, macOS),
        and on macOS a close right after the wake connection drops that
        connection from the queue, so `accept()` never returns (seen 2026-10-11
        on the M4: every second daemon of a test run hung). So this only wakes
        the loop, by one connection to its own address; the loop sees
        `_running` False, leaves, and closes its listener itself."""
        self._running = False
        sock = self._server_socket
        if sock is None:
            return
        try:
            addr = sock.getsockname()
            with socket.socket(sock.family, socket.SOCK_STREAM) as wake:
                wake.settimeout(self._settings["drain_timeout_s"])
                wake.connect(addr)
        except OSError as e:
            print(f"[Daemon] Listener already closed or unreachable at wake: {e}")

    def _handle_signal(self, signum, frame):
        """Graceful shutdown on SIGTERM/SIGINT."""
        sig_name = signal.Signals(signum).name
        print(f"\n[Daemon] Received {sig_name} — shutting down")
        self._ops.stop_all("shutdown")
        # Close server socket to unblock accept()
        self._close_listener()

    def _cleanup(self) -> None:
        """Clean up socket and PID files."""
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError as e:
                print(f"[Daemon] Listener close: {e}")

        # The instance's files: socket (Unix), pid, port (named Windows instance)
        for f in INSTANCE.instance_files():
            if f.exists():
                try:
                    f.unlink()
                except OSError as e:
                    print(f"[Daemon] Could not remove {f}: {e}")

        print("[Daemon] Cleaned up")


class _ForkedChild:
    """The `poll()` of a forked child, so `wait_ready` reads its exit code."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: Optional[int] = None

    def poll(self) -> Optional[int]:
        if self.returncode is None:
            done, status = os.waitpid(self.pid, os.WNOHANG)
            if done:
                self.returncode = os.waitstatus_to_exitcode(status)
        return self.returncode

    def wait(self) -> int:
        if self.returncode is None:
            self.returncode = os.waitstatus_to_exitcode(os.waitpid(self.pid, 0)[1])
        return self.returncode


def wait_ready(proc, read_fd: Optional[int] = None, ready_file=None,
               poll_s: Optional[float] = None) -> Dict[str, Any]:
    """The launcher's side: the daemon's ready record, or a failed record
    naming the exit code of a child that ended without one.

    With `read_fd` (POSIX) it blocks on the pipe the child inherited: no
    sleep, no deadline — the child's exit closes the pipe. With `ready_file`
    (Windows) it watches the file every `poll_s` until the file appears or
    the child exits; the file is removed once read."""
    if read_fd is not None:
        with os.fdopen(read_fd, "r") as pipe:
            line = pipe.readline()
        if line.strip():
            return json.loads(line)
        code = proc.wait()
    else:
        path = Path(ready_file)
        if poll_s is None:
            poll_s = serving_settings()["ready_poll_s"]
        while True:
            if path.exists():
                record = json.loads(path.read_text())
                path.unlink()
                return record
            code = proc.poll()
            if code is not None:
                if path.exists():
                    continue   # written just before the exit: read it
                break
            time.sleep(poll_s)
    return {"state": "failed", "pid": proc.pid,
            "error": f"the daemon process exited with code {code} before it was ready; see {LOG_PATH}"}


def _announce(record: Dict[str, Any]) -> bool:
    """What `neurobrix serve` prints once its background daemon answered."""
    if record.get("state") != "ready":
        print(f"[Serve] Failed: {record.get('error')}")
        return False
    print(f"[Serve] Model loaded — {record['model']} ({record['family']}, {record['serving']} serving)")
    print(f"[Serve] Use 'neurobrix chat' to connect")
    print(f"[Serve] Use 'neurobrix stop' to shutdown")
    return True
