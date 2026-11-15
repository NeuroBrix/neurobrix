"""THE one reader of a container's processor declaration (`manifest.json`, top-level key `processors`).

The container's builder writes the declaration: one record per processor, with the EXACT artifact a model runs (its
file name, size and sha256) and the hosts that serve it, named by key, never by URL. This module reads it and names
every shape it refuses; it decides nothing about fetching (store.py) or calling (interface.py). No key names a model
or a tool: a processor's name is a value.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

#: the manifest key (open question F8: an added key before format 0.2) and the artifact's extension (F9)
MANIFEST_KEY = "processors"
ARTIFACT_SUFFIX = ".nbxp"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TEXT = ("name", "kind", "label", "version", "file", "sha256")
_LISTS = ("variants", "licences", "sources")


class DeclarationRefused(ValueError):
    """The container's processor declaration has a shape the engine cannot read: the container is to be built again."""


@dataclass(frozen=True)
class Declaration:
    """One declared processor, as the container states it."""
    name: str
    kind: str
    label: str
    version: str
    interface: int
    file: str
    size: int
    sha256: str
    variants: Tuple[str, ...]
    licences: Tuple[str, ...]
    sources: Tuple[str, ...]
    entry: bool = False
    uses: Dict[str, str] = field(default_factory=dict)
    components: Dict[str, str] = field(default_factory=dict)


def _refuse(where: str, what: str) -> DeclarationRefused:
    return DeclarationRefused(f"The container's manifest key {MANIFEST_KEY}{where} {what}: the container cannot be "
                              f"read by this NeuroBrix. Import the model again; if it stays, the container is to be "
                              f"built again.")


def _texts(rec: dict, where: str) -> dict:
    """The record's text fields, each present and non-empty; the file a bare name; the sha256 64 lower hex digits."""
    out = {}
    for key in _TEXT:
        value = rec.get(key)
        if not isinstance(value, str) or not value:
            raise _refuse(where, f"has no text '{key}'")
        out[key] = value
    if "/" in out["file"] or "\\" in out["file"] or out["file"].startswith("."):
        raise _refuse(where, f"names the file {out['file']!r}, which is not a bare file name")
    if not _SHA256.match(out["sha256"]):
        raise _refuse(where, "has a 'sha256' that is not 64 lower-case hex digits")
    return out


def _numbers(rec: dict, where: str) -> dict:
    """`interface` and `size`: positive integers (a bool is not one)."""
    out = {}
    for key in ("interface", "size"):
        value = rec.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise _refuse(where, f"has no positive integer '{key}'")
        out[key] = value
    return out


def _lists(rec: dict, where: str) -> dict:
    """`variants`, `licences`, `sources`: non-empty lists of non-empty texts."""
    out = {}
    for key in _LISTS:
        value = rec.get(key)
        if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
            raise _refuse(where, f"has no non-empty list of texts '{key}'")
        out[key] = tuple(value)
    return out


def _maps(rec: dict, where: str) -> dict:
    """`uses` and `components`: optional maps of text to text; `entry` an optional boolean."""
    out = {}
    for key in ("uses", "components"):
        value = rec.get(key, {})
        if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) and k and v
                                                  for k, v in value.items()):
            raise _refuse(where, f"has a '{key}' that is not a map of names")
        out[key] = dict(value)
    if not isinstance(rec.get("entry", False), bool):
        raise _refuse(where, "has an 'entry' that is not true or false")
    out["entry"] = rec.get("entry", False)
    return out


def _one(rec, index: int) -> Declaration:
    where = f"[{index}]"
    if not isinstance(rec, dict):
        raise _refuse(where, "is not a record")
    return Declaration(**_texts(rec, where), **_numbers(rec, where), **_lists(rec, where), **_maps(rec, where))


def _whole(decls: List[Declaration]) -> None:
    """Across records: names and files unique, every `uses` target declared, exactly one entry when several."""
    names = [d.name for d in decls]
    for key in ("name", "file"):
        seen = [getattr(d, key) for d in decls]
        if len(set(seen)) != len(seen):
            raise _refuse("", f"declares the same {key} twice")
    for d in decls:
        missing = sorted(set(d.uses.values()) - set(names))
        if missing:
            raise _refuse(f" ({d.name})", f"uses {', '.join(missing)}, which it does not declare")
    if len(decls) > 1 and sum(d.entry for d in decls) != 1:
        raise _refuse("", "declares several processors and not exactly one 'entry'")


def declared(manifest: dict) -> List[Declaration]:
    """The processors `manifest` declares, in their order; an empty list when the key is absent."""
    if MANIFEST_KEY not in manifest:
        return []
    records = manifest[MANIFEST_KEY]
    if not isinstance(records, list) or not records:
        raise _refuse("", "is not a non-empty list")
    decls = [_one(rec, i) for i, rec in enumerate(records)]
    _whole(decls)
    return decls


def entry_of(decls: List[Declaration]) -> Declaration:
    """The processor the engine calls: the only one declared, or the one marked `entry`."""
    return decls[0] if len(decls) == 1 else next(d for d in decls if d.entry)
