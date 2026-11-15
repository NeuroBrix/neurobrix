"""A toy processor artifact, a toy model that declares it, and a local host that serves it: what the processors'
tests and the recorded session drive the real door with. No real processor is here; the toy reads letters."""
from __future__ import annotations

import hashlib
import http.server
import io
import json
import threading
import time
import zipfile
from pathlib import Path

NAME, VERSION, LABEL = "toy_letters", "0.0.1+0", "toy letters"
FILE = f"{NAME}-{VERSION}.nbxp"
FIXED_TIME = (2026, 1, 1, 0, 0, 0)

READER = '''
class FrontEnd:
    """Each letter of the text becomes the unit its table names."""

    def __init__(self, data):
        self.table = dict(line.split("\\t") for line in data.text("letters.tsv").splitlines())

    def count(self, text, variant):
        return len(self.units(text, variant, None))

    def units(self, text, variant, ask):
        return [self.table.get(c, c) + ("" if variant == "en-us" else "'") for c in text.lower() if c.isalpha()]
'''


def artifact(padding: int = 0) -> bytes:
    """The toy artifact's bytes: a deterministic zip (sorted names, fixed times); `padding` bytes of data make a
    download long enough to watch."""
    members = {
        "processor.json": json.dumps({"name": NAME, "version": VERSION, "kind": "text_front_end", "interface": 1,
                                      "entry": "reader:FrontEnd"}, sort_keys=True).encode(),
        "code/__init__.py": b"",
        "code/reader.py": READER.encode(),
        "data/letters.tsv": "\n".join(f"{c}\t{c.upper()}" for c in "abcdefghijklmnopqrstuvwxyz").encode(),
    }
    if padding:
        members["data/padding.bin"] = bytes(i % 251 for i in range(padding))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:
        for name in sorted(members):
            zf.writestr(zipfile.ZipInfo(name, FIXED_TIME), members[name])
    return out.getvalue()


def declaration(data: bytes, sources=("hub", "hf"), **over) -> dict:
    """The container's record for the toy artifact `data`."""
    rec = {"name": NAME, "kind": "text_front_end", "label": LABEL, "version": VERSION, "min_version": "0.0.1",
           "interface": 1, "variants": ["en-us", "en-gb"], "file": FILE, "size": len(data),
           "sha256": hashlib.sha256(data).hexdigest(), "licences": ["Apache-2.0"], "sources": list(sources)}
    rec.update(over)
    return rec


def install_model(cache: Path, model: str, processors=None) -> Path:
    """A toy model's cache entry: a manifest, with the `processors` key when `processors` is given."""
    entry = cache / model
    entry.mkdir(parents=True, exist_ok=True)
    manifest = {"model_name": model, "family": "tts", "format_version": "0.1"}
    if processors is not None:
        manifest["processors"] = processors
    (entry / "manifest.json").write_text(json.dumps(manifest))
    from neurobrix.nbx.names import record_name
    record_name(entry, hub_name(model))
    return entry


def hub_name(model: str) -> str:
    """The name a person types for the toy model `model` (recorded beside its manifest, as an import records it)."""
    return f"toy/{model}"


class Host:
    """A local host serving `files` (name -> bytes) under `/processors/`, counting the GETs per name; `rate` bytes a
    second when given (a download to watch); `status` a forced reply code."""

    def __init__(self, files: dict, rate: int = 0, status: int = 0):
        self.files, self.rate, self.status, self.gets = dict(files), rate, status, {}
        host = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                return None

            def do_GET(self):
                name = self.path.rsplit("/", 1)[-1]
                host.gets[name] = host.gets.get(name, 0) + 1
                body = host.files.get(name)
                if host.status or body is None:
                    self.send_response(host.status or 404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                step = max(1, host.rate // 10) if host.rate else len(body)
                for i in range(0, len(body), step):
                    self.wfile.write(body[i:i + step])
                    if host.rate:
                        time.sleep(0.1)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def dead_url() -> str:
    """A local address nothing listens on: a host that does not answer."""
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


#: the toy processor's source sub-package, as tools/build_processor.py reads it (its data is served by a Host)
SOURCE_DATA = {"letters.tsv": "\n".join(f"{c}\t{c.upper()}" for c in "abcdefghijklmnopqrstuvwxyz").encode(),
               "COPYING": b"The toy data's licence text.\n"}


def write_source(root: Path, data_url: str, reader: str = READER, padding: int = 0) -> Path:
    """The toy processor's source sub-package under `root` (its name the folder's), its data pinned at `data_url`
    (a Host's `/processors/` folder); `padding` bytes of pinned data make a download long enough to watch."""
    data = source_data(padding)
    sub = root / NAME
    (sub / "reader").mkdir(parents=True, exist_ok=True)
    (sub / "reader" / "__init__.py").write_text("")
    (sub / "reader" / "reader.py").write_text(reader)
    (sub / "LICENSE").write_text("Apache-2.0, the toy reader's licence text.\n")
    (sub / "EQUALITY.md").write_text("The toy reader against its table: 26 / 26 letters equal.\n")
    pins = [{"path": path, "url": f"{data_url}/processors/{path}", "revision": "toy-1",
             "sha256": hashlib.sha256(body).hexdigest(), "size": len(body)} for path, body in sorted(data.items())]
    rec = {"name": NAME, "version": VERSION, "kind": "text_front_end", "interface": 1, "entry": "reader:FrontEnd",
           "variants": ["en-us", "en-gb"], "code": {"source": "reader", "licence": "Apache-2.0"},
           "data": {"licence": "LicenseRef-toy", "licence_file": "COPYING", "files": pins}}
    (sub / "processor.json").write_text(json.dumps(rec, indent=2))
    return sub


def source_data(padding: int = 0) -> dict:
    """The files a Host serves for the toy source's pins."""
    data = dict(SOURCE_DATA)
    if padding:
        data["padding.bin"] = bytes(i % 251 for i in range(padding))
    return data
