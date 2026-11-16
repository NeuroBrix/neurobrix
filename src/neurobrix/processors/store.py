"""The door of the processors: find an artifact in the cache, fetch it, verify its sha256, load it; install a file.

The trust of a processor's code rests on ONE value: the sha256 the model's container configuration declares. The
artifact is read into memory once, those bytes are hashed, and only those bytes are ever executed or read: nothing
is unpacked, nothing enters the import path before the checksum holds, and a file in the cache is verified again at
every process start that needs it. Its code is imported from the verified bytes under a module name that carries the
sha256, so two versions live side by side and a processor's own name is never imported.

The cache is `<the NeuroBrix cache>/processors/<file>`, one file per artifact, never unpacked. Bytes arrive beside it
and take its name only once the sha256 holds. Two first runs at once: the second waits for the first's lock, then
finds the verified file and says nothing.

Hosts are tried in the declared order; the next one only when the first does not ANSWER (no connection, no reply in
time, a 5xx). A checksum mismatch is final: no other host, no retry.

This is the only module of the processors that touches the network, the hash and the lock; it is imported only when
a container declares a processor.
"""
from __future__ import annotations

import hashlib
import importlib.abc
import importlib.util
import io
import json
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from neurobrix.processors import messages
from neurobrix.processors.declaration import ARTIFACT_SUFFIX, Declaration, entry_of
from neurobrix.processors.interface import (Ask, ArtifactData, Held, refuse_missing_components,
                                            refuse_unspoken)

log = logging.getLogger("neurobrix.processors")

#: how often a run waiting for another run's download looks at the lock again
LOCK_POLL_SECONDS = 0.2
#: where a declared source key serves an artifact (open question F5: the hub, then Hugging Face, both declared by the
#: container in its order); the hosts are the engine's own settings, never a URL from a container
SOURCE_PATHS = {"hub": "{hub}/processors/{file}", "hf": "{hf}/Neurobrix/processors/resolve/main/{file}"}


def _hosts() -> Dict[str, str]:
    """The engine's own host settings: the registry the import command reads, and the Hugging Face endpoint."""
    from neurobrix.cli.utils import REGISTRY_URL
    return {"hub": REGISTRY_URL.rstrip("/"), "hf": (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")}


def source_urls(decl: Declaration) -> List[str]:
    """The declared sources of `decl`, in order, as the URLs of its file; a source this engine does not know is
    refused (a newer engine reads it)."""
    hosts = _hosts()
    for key in decl.sources:
        if key not in SOURCE_PATHS:
            raise messages.refusal(messages.TOO_NEW, label=decl.label)
    return [SOURCE_PATHS[key].format(file=decl.file, **hosts) for key in decl.sources]


def cache_root() -> Path:
    """The processors' folder under the NeuroBrix cache setting."""
    from neurobrix.core.paths import cache_dir
    return cache_dir() / "processors"


def verified(path: Path, decl: Declaration) -> Optional[bytes]:
    """The bytes of `path` when their sha256 is the declared one; None when there is no file."""
    if not path.is_file():
        return None
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != decl.sha256:
        path.unlink()
        raise messages.refusal(messages.MISMATCH, label=decl.label)
    return data


def _announced(decl: Declaration):
    """The bar factory the download brick calls once a host answers: the first line of the first run, then the
    import command's bar (tqdm: size, speed, time left) on a terminal, and no bar on a pipe, CI or a daemon's log."""
    from neurobrix.cli.utils import format_size

    def bar(**columns):
        log.info(messages.BEFORE.format(label=decl.label, size=format_size(decl.size)))
        from tqdm import tqdm
        return tqdm(disable=not sys.stderr.isatty(), **columns)
    return bar


def _download(decl: Declaration, arrived: Path, model: str) -> None:
    """The declared sources tried in order into `arrived`; the next only when one does not answer."""
    import requests
    from neurobrix.cli.commands.registry import IncompleteDownload, download_resumable
    cause = None
    for url in source_urls(decl):
        try:
            download_resumable(url, arrived, total_hint=decl.size, desc=decl.label, progress=_announced(decl))
            return
        except (requests.ConnectionError, requests.Timeout) as exc:
            cause = exc
        except requests.HTTPError as exc:
            if exc.response is None or exc.response.status_code < 500:
                raise messages.refusal(messages.UNPUBLISHED, label=decl.label, name=decl.name,
                                       version=decl.version) from exc
            cause = exc
        except IncompleteDownload as exc:
            raise messages.refusal(messages.CUT, label=decl.label) from exc
    raise messages.refusal(messages.UNREACHABLE, label=decl.label, model=model, file=decl.file) from cause


def _fetch(decl: Declaration, dest: Path, model: str) -> bytes:
    """Download `decl` beside `dest`, verify it, give it `dest`'s name; the ready line once it holds."""
    arrived = dest.with_name(dest.name + ".arrived")
    _download(decl, arrived, model)
    data = arrived.read_bytes()
    if hashlib.sha256(data).hexdigest() != decl.sha256:
        arrived.unlink()
        raise messages.refusal(messages.MISMATCH, label=decl.label)
    os.replace(arrived, dest)
    log.info(messages.READY.format(label=decl.label))
    return data


def ensure(decl: Declaration, model: str, root: Optional[Path] = None) -> bytes:
    """The verified bytes of `decl`'s artifact in `root` (the cache by default): found there in silence, or fetched
    once under the lock beside it."""
    from neurobrix.nbx.atomic_install import held_in_turn
    dest = (root or cache_root()) / decl.file
    data = verified(dest, decl)
    if data is not None:
        return data
    with held_in_turn(dest.with_name(dest.name + ".lock"), label=f"fetch {decl.file}", poll_seconds=LOCK_POLL_SECONDS):
        data = verified(dest, decl)
        return data if data is not None else _fetch(decl, dest, model)


class _VerifiedCode(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Imports the modules of one artifact's `code/` tree from its verified bytes, under `prefix` alone."""

    def __init__(self, prefix: str, archive: zipfile.ZipFile, origin: str):
        self.prefix, self.archive, self.origin = prefix, archive, origin
        self.members = set(archive.namelist())

    def find_spec(self, name, path=None, target=None):
        if name != self.prefix and not name.startswith(self.prefix + "."):
            return None
        rel = name[len(self.prefix) + 1:].replace(".", "/")
        package = f"code/{rel}/__init__.py" if rel else "code/__init__.py"
        if package in self.members or not rel:
            return importlib.util.spec_from_loader(name, self, origin=f"{self.origin}/{package}", is_package=True)
        if f"code/{rel}.py" in self.members:
            return importlib.util.spec_from_loader(name, self, origin=f"{self.origin}/code/{rel}.py")
        return None

    def exec_module(self, module):
        member = module.__spec__.origin[len(self.origin) + 1:]
        source = self.archive.read(member) if member in self.members else b""
        exec(compile(source, module.__spec__.origin, "exec"), module.__dict__)


def _manifest_of(archive: zipfile.ZipFile, decl: Declaration) -> dict:
    """The artifact's own `processor.json`, which names what the container's declaration names."""
    own = json.loads(archive.read("processor.json"))
    for key in ("name", "version", "kind", "interface"):
        if own.get(key) != getattr(decl, key):
            raise messages.refusal(messages.MISMATCH, label=decl.label)
    return own


def load(decl: Declaration, data: bytes):
    """The processor object of a verified artifact: its `entry` (`module:Class`) built with its `data/` tree."""
    archive = zipfile.ZipFile(io.BytesIO(data))
    own = _manifest_of(archive, decl)
    prefix = f"_nbxp_{decl.sha256[:16]}"
    if prefix not in sys.modules:
        sys.meta_path.insert(0, _VerifiedCode(prefix, archive, decl.file))
    module_name, _, class_name = own["entry"].partition(":")
    module = importlib.import_module(f"{prefix}.{module_name}")
    return getattr(module, class_name)(ArtifactData(archive))


def open_all(decls: List[Declaration], components, model: str) -> Held:
    """Every declared processor checked, made present, verified and loaded; the entry one returned, bound (no runner
    of a declared component is lent yet: `Ask.labels` refuses by its sentence)."""
    for decl in decls:
        refuse_unspoken(decl)
        source_urls(decl)
        refuse_missing_components(decl, components)
    loaded = {d.name: load(d, ensure(d, model)) for d in decls}
    held: Dict[str, Held] = {}

    def bind(decl: Declaration) -> Held:
        if decl.name not in held:
            used = {role: bind(next(d for d in decls if d.name == target)) for role, target in decl.uses.items()}
            held[decl.name] = Held(decl, loaded[decl.name], Ask(decl, used, None))
        return held[decl.name]
    return bind(entry_of(decls))


def install(file: Path, decls: Optional[List[Declaration]] = None, model: str = "") -> Path:
    """Copy `file` into the cache under its lock: checked at once against `decls` when given (the ready line said),
    else stored under its name and checked at the first run that declares it."""
    from neurobrix.nbx.atomic_install import held_in_turn
    if not file.is_file():
        raise messages.refusal(messages.NOT_A_FILE, path=file)
    decl = None
    if decls is not None:
        decl = next((d for d in decls if d.file == file.name), None)
        if decl is None:
            raise messages.refusal(messages.NOT_DECLARED, file=file.name, model=model,
                                   files=", ".join(d.file for d in decls))
    dest = cache_root() / file.name
    with held_in_turn(dest.with_name(dest.name + ".lock"), label=f"install {file.name}", poll_seconds=LOCK_POLL_SECONDS):
        staged = dest.with_name(dest.name + ".arrived")
        shutil.copyfile(file, staged)
        if decl is not None and hashlib.sha256(staged.read_bytes()).hexdigest() != decl.sha256:
            staged.unlink()
            raise messages.refusal(messages.MISMATCH, label=decl.label)
        os.replace(staged, dest)
    if decl is not None:
        log.info(messages.READY.format(label=decl.label))
    return dest


def listing() -> List[dict]:
    """The artifacts in the cache: file, size, sha256."""
    root = cache_root()
    files = sorted(root.glob("*" + ARTIFACT_SUFFIX)) if root.is_dir() else []
    return [{"file": p.name, "size": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in files]
