"""A model's ONE name: the hub name `neurobrix import <org>/<name>` recorded beside its cache entry.

The container carries the name it was built under (`manifest.model_name`, the cache entry's directory); the model's
public name is its hub name (its Hugging Face or vendor repository's), which only the hub record knows. `import`
records it in `<entry>/name.json` — when it installs the entry, and when it finds the entry already installed — and
every user-facing place reads it from there: `neurobrix list`, `--model`, the API's model id.

An entry has exactly one name at any time:
  * recorded  — its hub name; the directory name is not accepted as a second one (refused, naming the hub name);
  * no record — an entry installed before 0.6.3 is still named by its directory, and every use says how to record its
                hub name (`MIGRATION_NOTE`). That is a migration message, not an alias.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator, Optional, Tuple

RECORD = "name.json"


class NameRefused(LookupError):
    """A name that is not the entry's one name (the entry carries a recorded hub name): says both."""

    def __init__(self, asked: str, recorded: str) -> None:
        super().__init__(f"the model installed as {asked} is named {recorded}: use --model {recorded}")
        self.asked, self.recorded = asked, recorded


def migration_note(entry_name: str, hub_name: Optional[str] = None) -> str:
    return (f"{entry_name} was installed before 0.6.3: run `neurobrix import {hub_name or '<org>/' + entry_name}` "
            f"once to record its name")


def recorded_name(entry: Path) -> Optional[str]:
    """The hub name recorded beside the entry's manifest, or None (installed before 0.6.3). A record that is not
    the shape `record_name` writes is refused by name, never read as absent."""
    path = Path(entry) / RECORD
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        name = data["name"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise RuntimeError(f"ZERO FALLBACK: the name record {path} is unreadable ({e}); `neurobrix import "
                           f"<org>/<name>` writes it again.") from e
    if not isinstance(name, str) or "/" not in name:
        raise RuntimeError(f"ZERO FALLBACK: the name record {path} holds {name!r}, not an <org>/<name> hub name.")
    return name


def record_name(entry: Path, name: str, container_sha256: Optional[str] = None) -> None:
    """Write the entry's hub name, whole (written beside, then renamed over), with the sha256 of the container it was
    installed from when the hub states it: a later import finds the entry by that sha before any byte is fetched."""
    if not isinstance(name, str) or "/" not in name:
        raise ValueError(f"a hub name is <org>/<name>, got {name!r}")
    from neurobrix import __version__
    entry = Path(entry)
    tmp = entry / f".{RECORD}.{os.getpid()}"
    rec = {"name": name, "recorded_by": f"neurobrix {__version__} import"}
    if container_sha256:
        rec["container_sha256"] = container_sha256
    tmp.write_text(json.dumps(rec) + "\n")
    os.replace(tmp, entry / RECORD)


def recorded_container_sha256(entry: Path) -> Optional[str]:
    """The sha256 of the container the entry was installed from, as its record states it; None when not recorded."""
    path = Path(entry) / RECORD
    if not path.exists():
        return None
    recorded_name(entry)                             # an unreadable record is refused by name there
    return json.loads(path.read_text()).get("container_sha256")


def entry_of_container(cache: Path, sha256: str) -> Optional[Path]:
    """The installed entry whose record names the container `sha256`, or None."""
    for entry in _entries(Path(cache)):
        if sha256 and recorded_container_sha256(entry) == sha256:
            return entry
    return None


def _entries(cache: Path) -> Iterator[Path]:
    from neurobrix.nbx.atomic_install import is_install_artifact
    if not cache.exists():
        return
    for d in sorted(cache.iterdir()):
        if d.is_dir() and not is_install_artifact(d.name) and (d / "manifest.json").exists():
            yield d


def name_of(cache: Path, entry_name: str) -> str:
    """The one name of the entry `entry_name`: its recorded hub name, else (no record) its directory name."""
    entry = Path(cache) / entry_name
    rec = recorded_name(entry) if entry.is_dir() else None
    return rec if rec is not None else entry_name


def accepted_names(cache: Path) -> list:
    """The name `--model` accepts for each installed entry, sorted: what a "not found" answer lists."""
    return sorted(name_of(cache, entry.name) for entry in _entries(Path(cache)))


def resolve(cache: Path, asked: str) -> Tuple[str, Optional[str]]:
    """The cache entry a user's model name designates, and the migration note to say (None when the name is
    recorded). A recorded entry asked by another name that leads to it — its directory name, bare or behind the recorded
    name's own organisation (`import` takes names the hub answers under another one: `<org>/Sana_1600M_1024px_MultiLing` is
    recorded as `<org>/Sana_1600M_1024px_MultiLing_diffusers`) — is refused (`NameRefused`) naming its hub name: one
    message, whatever was typed. A name no entry carries is returned as asked, for the caller's "not found" answer."""
    cache = Path(cache)
    for entry in _entries(cache):
        if recorded_name(entry) == asked:
            return entry.name, None
    org, entry = asked.rpartition("/")[0], cache / asked.rpartition("/")[2]
    if entry.is_dir() and (entry / "manifest.json").exists():
        rec = recorded_name(entry)
        if rec is not None and org in ("", rec.rpartition("/")[0]):     # another organisation's model is another model
            raise NameRefused(asked, rec)
        if rec is None and not org:
            return asked, migration_note(asked)
    return asked, None


def container_files(entry: Path):
    """The files of a cache entry that belong to its CONTAINER, sorted: every file but the local name record, which
    this machine's `import` wrote beside the manifest and which no `.nbx` carries. A tool that verifies an entry
    against its hub object, repacks it, or fingerprints it reads this list, never the raw tree."""
    entry = Path(entry)
    record = entry / RECORD
    return sorted(p for p in entry.rglob("*") if p.is_file() and p != record and not p.name.startswith(f".{RECORD}."))
