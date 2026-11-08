"""The HTTP door: an OpenAI-compatible server on the SAME daemon as the Unix socket.

`ThreadingHTTPServer` (stdlib): one thread per connection, which only parses, submits the work to the daemon's one
inference worker, waits and writes the answer back — the dispatcher's shape (serving/server.py). The engine call
blocks, so an asyncio loop would only add a hop to an executor.

Where it listens: `serving.http_host` / `serving.http_port` (config/system.yml) unless `serve --http-host` /
`--http-port` say otherwise. A non-loopback address without an API key is refused at start; with a key, every
request carries `Authorization: Bearer <key>` (compared in constant time). A port already taken is refused naming the
process that holds it.

Streaming is Server-Sent Events: one `data: <json>` line per chunk, then `data: [DONE]`; the response is
close-delimited (no length), as OpenAI's own.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from email.parser import BytesParser
from email.policy import HTTP as _HTTP_POLICY
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from neurobrix.serving.openai_api import errors, routes

_log = logging.getLogger(__name__)

_HTTP_SETTINGS = {"http_host": (str,), "http_port": (int,), "http_max_queued": (int,), "http_max_body_mb": (int,)}


def http_settings() -> Dict[str, Any]:
    """The door's settings, from `serving:` in config/system.yml; a missing or invalid one is refused by name."""
    import yaml
    from neurobrix.core.workspace import SYSTEM_YML
    cfg = (yaml.safe_load(SYSTEM_YML.read_text()) or {}).get("serving") or {}
    out = {}
    for key, types in _HTTP_SETTINGS.items():
        v = cfg.get(key)
        bad = isinstance(v, bool) or not isinstance(v, types) or (isinstance(v, int) and v <= 0) or v == ""
        if bad:
            raise RuntimeError(f"ZERO FALLBACK: the HTTP door needs `serving.{key}` in {SYSTEM_YML}, got {v!r}.")
        out[key] = v
    return out


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def read_api_key(path: Optional[str]) -> Optional[str]:
    """The API key, from a file (never from the command line, where `ps` shows it). An empty file is refused."""
    if path is None:
        return None
    text = Path(path).read_text().strip()
    if not text:
        raise RuntimeError(f"ZERO FALLBACK: the API key file {path} is empty.")
    return text


def port_owner(port: int) -> str:
    """Who holds a TCP port, read from the system: `lsof` where it exists, else Linux's /proc. Said, never guessed."""
    lsof = shutil.which("lsof")
    if lsof:
        try:
            out = subprocess.run([lsof, "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fpc"], capture_output=True,
                                 text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError) as e:
            return f"its owner could not be read ({e})"
        pid = cmd = None
        for line in out.splitlines():
            if line.startswith("p"):
                pid = line[1:]
            elif line.startswith("c"):
                cmd = line[1:]
        if pid:
            return f"held by {cmd or '?'} (PID {pid})"
    if sys.platform.startswith("linux"):
        inodes = set()
        for table in ("/proc/net/tcp", "/proc/net/tcp6"):
            try:
                rows = Path(table).read_text().splitlines()[1:]
            except OSError:
                continue
            for row in rows:
                cols = row.split()
                if len(cols) > 9 and cols[3] == "0A" and int(cols[1].rsplit(":", 1)[1], 16) == port:
                    inodes.add(cols[9])
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                for fd in (proc / "fd").iterdir():
                    link = os.readlink(fd)
                    if link.startswith("socket:[") and link[8:-1] in inodes:
                        comm = (proc / "comm").read_text().strip()
                        return f"held by {comm} (PID {proc.name})"
            except OSError:
                continue
        if inodes:
            return "held by a process this user cannot see"
    return "its owner could not be read (no lsof, no /proc)"


class _SSE:
    """The event stream of one response, written by the connection thread."""

    def __init__(self, handler: "_Handler") -> None:
        self._h = handler
        self.gone = False

    def send(self, obj: Any) -> None:
        if self.gone:
            return
        data = b"data: " + json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n\n"
        try:
            self._h.wfile.write(data)
            self._h.wfile.flush()
        except OSError as e:
            self._vanished(e)

    def guard(self, fn, *args) -> None:
        """Run a per-token delivery; a client gone stops the operation at the next token (an orphan)."""
        if self.gone:
            return
        try:
            fn(*args)
        except OSError as e:
            self._vanished(e)

    def _vanished(self, why) -> None:
        self.gone = True
        op_id = self._h.op_id
        if op_id is not None:
            self._h.door.daemon._ops.request_stop(op_id, "orphan")
            _log.debug(f"[HTTP] Client of {op_id} vanished ({why}) — stopping it at the next token")

    def done(self) -> None:
        if not self.gone:
            try:
                self._h.wfile.write(b"data: [DONE]\n\n")
                self._h.wfile.flush()
            except OSError as e:
                self._vanished(e)
        self._h.close_connection = True


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "NeuroBrix"
    door: "HttpDoor" = None   # set on the subclass the door builds

    def log_message(self, fmt, *args) -> None:   # the daemon's log, not stderr lines per request
        _log.debug("[HTTP] " + fmt % args)

    # -- plumbing ----------------------------------------------------------------
    def requested_op_id(self) -> Optional[str]:
        return self.headers.get("X-NeuroBrix-Op-Id") or None

    def _send_json(self, status: int, obj: Any) -> None:
        payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        if self.op_id is not None:
            self.send_header("X-NeuroBrix-Op-Id", self.op_id)
        self.end_headers()
        self.wfile.write(payload)

    def _send_bytes(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if self.op_id is not None:
            self.send_header("X-NeuroBrix-Op-Id", self.op_id)
        self.end_headers()
        self.wfile.write(data)

    def start_sse(self) -> _SSE:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.flush()
        self._sse = _SSE(self)
        return self._sse

    def _authorized(self) -> bool:
        key = self.door.api_key
        if key is None:
            return True
        got = self.headers.get("Authorization") or ""
        scheme, _, token = got.partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(token.strip().encode(), key.encode())

    def _body(self) -> bytes:
        length = self.headers.get("Content-Length")
        if length is None:
            raise errors.ApiError(411, "a request body needs Content-Length", "invalid_request_error")
        n = int(length)
        bound = self.door.settings["http_max_body_mb"] * 1024 * 1024
        if n > bound:
            raise errors.ApiError(413, f"the body is {n} bytes; serving.http_max_body_mb allows {bound}",
                                  "invalid_request_error")
        return self.rfile.read(n)

    def _json(self) -> Dict[str, Any]:
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip()
        raw = self._body()
        if ctype != "application/json":
            raise errors.invalid(f"the body must be application/json, got {ctype or 'no Content-Type'}")
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise errors.invalid(f"the body is not JSON: {e}")
        if not isinstance(body, dict):
            raise errors.invalid("the body must be a JSON object")
        return body

    def _form(self) -> Dict[str, Any]:
        ctype = self.headers.get("Content-Type") or ""
        if not ctype.startswith("multipart/form-data"):
            raise errors.invalid(f"the body must be multipart/form-data, got {ctype or 'no Content-Type'}")
        raw = self._body()
        msg = BytesParser(policy=_HTTP_POLICY).parsebytes(
            b"Content-Type: " + ctype.encode("latin-1") + b"\r\n\r\n" + raw)
        if not msg.is_multipart():
            raise errors.invalid("the multipart body has no parts")
        form: Dict[str, Any] = {}
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if not name:
                continue
            payload = part.get_payload(decode=True) or b""
            filename = part.get_filename()
            form[name] = (filename, payload) if filename is not None else payload.decode("utf-8")
        return form

    # -- verbs --------------------------------------------------------------------
    def do_GET(self) -> None:
        self._serve("GET")

    def do_POST(self) -> None:
        self._serve("POST")

    def _serve(self, verb: str) -> None:
        self.op_id = None
        self._sse = None
        door = self.door
        door.activity(+1)
        try:
            if not self._authorized():
                raise errors.unauthorized()
            path = self.path.split("?", 1)[0].rstrip("/")
            if verb == "GET" and path == "/v1/models":
                return self._send_json(200, routes.models(door))
            if verb == "GET" and path.startswith("/v1/models/"):
                asked = path[len("/v1/models/"):]
                if asked != door.model_id():
                    raise errors.model_not_found(asked, door.model_id())
                return self._send_json(200, routes.model_record(door))
            if verb == "POST" and path == "/v1/chat/completions":
                out = routes.chat_completions(door, self, self._json())
            elif verb == "POST" and path == "/v1/completions":
                out = routes.completions(door, self, self._json())
            elif verb == "POST" and path == "/v1/images/generations":
                out = routes.images_generations(door, self, self._json())
            elif verb == "POST" and path == "/v1/audio/speech":
                return self._send_bytes(200, routes.audio_speech(door, self, self._json()), "audio/wav")
            elif verb == "POST" and path == "/v1/audio/transcriptions":
                out = routes.audio_transcriptions(door, self, self._form())
                if isinstance(out, str):
                    return self._send_bytes(200, out.encode("utf-8"), "text/plain; charset=utf-8")
            elif verb == "POST" and path == "/v1/embeddings":
                out = routes.embeddings(door, self, self._json())
            else:
                raise errors.ApiError(404, f"no route {verb} {path}", "invalid_request_error", None,
                                      "unknown_route")
            if out is not None:
                self._send_json(200, out)
        except errors.ApiError as e:
            if self._sse is not None:
                self._sse.send(e.body())
                self._sse.done()
            else:
                self._send_json(e.status, e.body())
        except OSError as e:
            _log.debug(f"[HTTP] connection lost: {e}")
            self.close_connection = True
        except Exception as e:   # an engine refusal or a defect: said, never a dropped connection
            _log.exception("[HTTP] request failed")
            err = errors.engine_failed(f"{type(e).__name__}: {e}")
            if self._sse is not None:
                self._sse.send(err.body())
                self._sse.done()
            else:
                try:
                    self._send_json(err.status, err.body())
                except OSError:
                    self.close_connection = True
        finally:
            door.activity(-1)


class HttpDoor:
    """The door of one daemon: bound at `open()`, served on its own accept thread, closed with the daemon."""

    def __init__(self, daemon, host: Optional[str] = None, port: Optional[int] = None,
                 api_key: Optional[str] = None, settings: Optional[Dict[str, Any]] = None) -> None:
        self.daemon = daemon
        self.settings = settings if settings is not None else http_settings()
        self.host = host if host is not None else self.settings["http_host"]
        self.port = int(port) if port is not None else self.settings["http_port"]
        self.api_key = api_key
        if not is_loopback(self.host) and api_key is None:
            raise RuntimeError(
                f"ZERO FALLBACK: the HTTP door on {self.host} (not a loopback address) needs an API key "
                f"(serve --api-key-file <file>): without one, anyone on the network drives the model.")
        from neurobrix.serving.protocol import DAEMON_DIR
        self.spool_root = Path(DAEMON_DIR) / "spool" / str(os.getpid())
        self.started = int(time.time())
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._active = 0
        self._last = time.monotonic()
        self._lock = threading.Lock()

    # -- what the routes ask ------------------------------------------------------
    def model_id(self) -> str:
        return self.daemon.served_name()

    def watch(self, handler: _Handler, op) -> None:
        """Notice a client that vanished while its non-streamed request runs: the daemon's own connection watch."""
        t = threading.Thread(target=self.daemon._watch_client, args=(handler.connection, op),
                             name=f"nbx-http-watch-{op.op_id}", daemon=True)
        t.start()

    def activity(self, delta: int) -> None:
        with self._lock:
            self._active += delta
            self._last = time.monotonic()

    def idle_for(self) -> float:
        """Seconds since the last request began or ended; 0 while one is open."""
        with self._lock:
            return 0.0 if self._active else time.monotonic() - self._last

    # -- life ---------------------------------------------------------------------
    def open(self) -> Dict[str, Any]:
        handler = type("NbxHandler", (_Handler,), {"door": self})
        try:
            server = ThreadingHTTPServer((self.host, self.port), handler, bind_and_activate=False)
            server.daemon_threads = True
            server.server_bind()
            server.server_activate()
        except OSError as e:
            raise RuntimeError(f"the HTTP door cannot listen on {self.host}:{self.port}: {e.strerror or e}; "
                               f"port {self.port} {port_owner(self.port)}. Choose another with --http-port.") from e
        self._server = server
        self.port = server.server_address[1]
        self.spool_root.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=server.serve_forever, name="nbx-http", daemon=True)
        self._thread.start()
        return self.endpoint()

    def endpoint(self) -> Dict[str, Any]:
        return {"host": self.host, "port": self.port, "base_url": f"http://{self.host}:{self.port}/v1",
                "auth": self.api_key is not None}

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        shutil.rmtree(self.spool_root, ignore_errors=True)
