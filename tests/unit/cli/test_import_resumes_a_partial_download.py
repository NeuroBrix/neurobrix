"""D-IMPORT-RESUMABLE-DOWNLOAD: a download that dies keeps its bytes and resumes.

Against a local HTTP server that honours Range, and one that does not; the cut
is injected by a server that closes the stream after K bytes. Seen saying yes
(a resume appends exactly the missing tail, bytes identical) and no (a short
stream never produces the final file; a server that ignores Range is not
appended to).

Run: PYTHONPATH=src python -m pytest tests/unit/cli/test_import_resumes_a_partial_download.py
"""
import hashlib
import http.server
import socketserver
import threading
from pathlib import Path

import pytest
import requests

from neurobrix.cli.commands.registry import IncompleteDownload, download_resumable, partial_path

PAYLOAD = bytes(range(256)) * 4096          # 1 MiB, every byte position distinct mod 256


def _server(range_support: bool, cut_after: int | None = None):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            start = 0
            rng = self.headers.get("Range")
            if rng and range_support:
                start = int(rng.split("=")[1].split("-")[0])
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD)-1}/{len(PAYLOAD)}")
            else:
                self.send_response(200)
            body = PAYLOAD[start:]
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if cut_after is not None:
                self.wfile.write(body[:cut_after]); self.wfile.flush()
                self.connection.close()          # the cut
                return
            self.wfile.write(body)
    srv = socketserver.TCPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/m.nbx"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_a_full_download_lands_under_its_final_name_only(tmp_path):
    srv, url = _server(range_support=True)
    try:
        dest = tmp_path / "m.nbx"
        assert download_resumable(url, dest, get=requests.get, chunk_size=65536) == len(PAYLOAD)
        assert _sha(dest) == hashlib.sha256(PAYLOAD).hexdigest()
        assert not partial_path(dest).exists()
    finally:
        srv.shutdown()


def test_a_cut_keeps_the_partial_and_never_writes_the_final_name(tmp_path):
    srv, url = _server(range_support=True, cut_after=300_000)
    try:
        dest = tmp_path / "m.nbx"
        with pytest.raises(IncompleteDownload):
            download_resumable(url, dest, get=requests.get, chunk_size=65536)
        assert not dest.exists(), "a fragment must never carry the container's name"
        part = partial_path(dest)
        assert part.exists() and 0 < part.stat().st_size < len(PAYLOAD)
    finally:
        srv.shutdown()


def test_a_resume_appends_exactly_the_missing_tail(tmp_path):
    dest = tmp_path / "m.nbx"; part = partial_path(dest)
    part.write_bytes(PAYLOAD[:123_457])                       # what an earlier run left
    srv, url = _server(range_support=True)
    try:
        download_resumable(url, dest, get=requests.get, chunk_size=65536)
        assert _sha(dest) == hashlib.sha256(PAYLOAD).hexdigest() and not part.exists()
    finally:
        srv.shutdown()


def test_a_server_without_range_support_is_read_from_zero_not_appended_to(tmp_path):
    dest = tmp_path / "m.nbx"; part = partial_path(dest)
    part.write_bytes(b"\xff" * 1000)                          # a partial the server cannot continue
    srv, url = _server(range_support=False)
    try:
        download_resumable(url, dest, get=requests.get, chunk_size=65536)
        assert _sha(dest) == hashlib.sha256(PAYLOAD).hexdigest(), "appending would have spliced two streams"
    finally:
        srv.shutdown()


def test_a_partial_longer_than_the_announced_size_is_dropped_not_trusted(tmp_path):
    dest = tmp_path / "m.nbx"; part = partial_path(dest)
    part.write_bytes(b"\x00" * (len(PAYLOAD) + 10))
    srv, url = _server(range_support=True)
    try:
        # the server answers 416 or 206 past the end in the wild; ours answers 206 with an
        # empty body, which the brick reads as a resume that delivered nothing — refused.
        try:
            download_resumable(url, dest, get=requests.get, chunk_size=65536)
        except IncompleteDownload:
            pass
        assert not dest.exists() or _sha(dest) == hashlib.sha256(PAYLOAD).hexdigest()
    finally:
        srv.shutdown()
