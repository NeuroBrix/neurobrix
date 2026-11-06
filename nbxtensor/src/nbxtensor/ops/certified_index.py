"""The compiled index of a certified file, per memory class: what a launch reads instead of the file's JSON.

The certified directory stays JSON — readable, contributed by pull request (the supervisor's decision, 2026-11-06 10:46).
Parsing and validating it cost every launch ~4 s on the Volta directory (126 MB: 2.9 s of json parsing, 1.2 s of
validation, measured 2026-11-06). The first launch after a change validates the file once (`autotune_certified.validate`)
and writes, per memory class, the entries that class is served: one binary file, never a pickle — a header, then each
key and its certification as length-prefixed bytes (the certification a small JSON object, parsed only when its key is
asked), then the SHA-256 of all that. Its name carries the FINGERPRINT of everything the validation read: the source
file's bytes, the validator's own code, the profile's tolerance for the dtype and the kernel's key and config spaces —
a change to any of them names another index, and the file is validated again. An index whose bytes do not hash to its
trailer is refused by name.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Dict, Iterator, Mapping, Optional

MAGIC = b"NBXCIDX1"
_HEAD = struct.Struct("<8sII")             # magic, memory class (GB), number of keys
_LEN = struct.Struct("<I")


def fingerprint(source: bytes, tolerance: Optional[float], tuner) -> str:
    """The SHA-256 (hex, 32 chars) of what validating `source` reads: its bytes, this validator's code, the tolerance,
    and the kernel's key arity, argument names and config parameter sets (`autotune_certified.validate`)."""
    from nbxtensor.ops import autotune_certified as A
    h = hashlib.sha256()
    h.update(hashlib.sha256(source).digest())
    h.update(Path(A.__file__).read_bytes())
    h.update(repr(tolerance).encode())
    spaces = sorted(sorted(c.kwargs) for c in getattr(tuner, "configs", None) or [])
    h.update(repr((len(getattr(tuner, "keys", []) or []), list(getattr(tuner, "arg_names", []) or []),
                   spaces)).encode())
    return h.hexdigest()[:32]


def directory() -> Path:
    """Where the indexes live: beside the machine's replay cache (`ops.autotune_cache`), read when used."""
    from nbxtensor.ops import autotune_cache
    return Path(autotune_cache._dir()) / "certified_index"


def path_for(source: Path, memory_gb: int, fp: str) -> Path:
    vendor, profile = source.parent.parent.name, source.parent.name
    return directory() / vendor / profile / f"{source.stem}.{int(memory_gb)}g.{fp}.idx"


def write(path: Path, memory_gb: int, entries: Mapping[str, dict]) -> None:
    """The index of `entries` ({key text: the certification served at this class}), written atomically."""
    body = bytearray(_HEAD.pack(MAGIC, int(memory_gb), len(entries)))
    for key in sorted(entries):
        k, v = key.encode(), json.dumps(entries[key], sort_keys=True, separators=(",", ":")).encode()
        body += _LEN.pack(len(k)) + k + _LEN.pack(len(v)) + v
    body += hashlib.sha256(body).digest()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_bytes(bytes(body))
    os.replace(tmp, path)


class IndexedEntries(Mapping):
    """{key text: certification} read from an index: the keys at open, a certification parsed when its key is asked."""

    def __init__(self, data: bytes, path: Path):
        if len(data) < _HEAD.size + 32 or hashlib.sha256(data[:-32]).digest() != data[-32:]:
            raise RuntimeError(f"nbxtensor: the certified index {path} is corrupt (its bytes do not hash to its "
                               f"trailer) — refused; delete it and the next launch validates its source again")
        magic, self.memory_gb, n = _HEAD.unpack_from(data, 0)
        if magic != MAGIC:
            raise RuntimeError(f"nbxtensor: {path} is not a certified index ({magic!r}) — refused")
        self._data, self._at, p = data, {}, _HEAD.size
        for _ in range(n):
            (kl,) = _LEN.unpack_from(data, p)
            key = data[p + 4:p + 4 + kl].decode()
            p += 4 + kl
            (vl,) = _LEN.unpack_from(data, p)
            self._at[key] = (p + 4, vl)
            p += 4 + vl
        self._parsed: Dict[str, dict] = {}

    def __getitem__(self, key: str) -> dict:
        got = self._parsed.get(key)
        if got is None:
            at, n = self._at[key]
            got = self._parsed[key] = json.loads(self._data[at:at + n])
        return got

    def __iter__(self) -> Iterator[str]:
        return iter(self._at)

    def __len__(self) -> int:
        return len(self._at)


def read(path: Path) -> Optional[IndexedEntries]:
    """The index at `path`, None when it does not exist; refused by name when it is corrupt."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    return IndexedEntries(data, path)
