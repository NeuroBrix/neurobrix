"""Where this machine keeps its containers and its extracted models.

ONE door. Before 2026-09-17 the answer was written in four places and reached by
two different mechanisms:

    cli/utils.py          STORE_DIR / CACHE_DIR, literals under ~/.neurobrix
    nbx/cache.py          DEFAULT_CACHE_DIR, a second literal for the same thing
    cli/commands/coverage.py     os.environ["NEUROBRIX_CACHE"], a third answer
    tools/stimulus_from_depth.py  the same env var again

`cli/utils.py` calls itself "single source of truth for paths" in its own
docstring. It was not one, and nothing said so.

RESOLUTION ORDER, and each step is deliberate:

  1. the environment (`NEUROBRIX_CACHE`, `NEUROBRIX_STORE`, `NEUROBRIX_HOME`) —
     one run, one shell, an operator override;
  2. `~/.neurobrix/paths.json` — this MACHINE's durable answer, which is what a
     daemon and a cron job read too, and what a symlink would otherwise have to
     pretend;
  3. `~/.neurobrix/{cache,store}` — the default, unchanged.

A CONFIGURED path that does not exist is REFUSED BY NAME. It is not created and
it is not quietly replaced by the default: a machine told to read its models
from a mount that is not mounted must stop, or it silently re-extracts 6 GB onto
a full disk and calls that normal. The DEFAULT path is created, because there
the engine owns the location.

Why a file and not a symlink: a symlink moves the data for every reader of that
path, including things that are not this engine, and it cannot be read back to
say what the configuration IS. `neurobrix info` prints these values.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

#: Where the machine's durable answer lives. Not configurable — it is the thing
#: that says where everything else is.
CONFIG_FILE = Path.home() / ".neurobrix" / "paths.json"

_ENV = {"home": "NEUROBRIX_HOME", "cache": "NEUROBRIX_CACHE", "store": "NEUROBRIX_STORE"}


class PathNotConfigured(RuntimeError):
    """A configured location does not exist. Named, so a caller can tell it
    from an ordinary missing directory it may create."""


def _configured() -> dict:
    try:
        text = CONFIG_FILE.read_text()
    except FileNotFoundError:
        return {}
    except OSError as exc:                                # pragma: no cover
        raise PathNotConfigured(
            f"{CONFIG_FILE} exists and cannot be read ({exc}). The engine will "
            f"not guess where its models are.") from exc
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise PathNotConfigured(
            f"{CONFIG_FILE} is not valid JSON ({exc}). Repair it or remove it; "
            f"the engine will not fall back to a default it was told not to "
            f"use.") from exc
    if not isinstance(data, dict):
        raise PathNotConfigured(f"{CONFIG_FILE} must hold an object, got {type(data).__name__}")
    return data


def _resolve(key: str, default: Path, *, create: bool) -> Path:
    env = os.environ.get(_ENV[key])
    if env:
        return _checked(Path(env).expanduser(), f"${_ENV[key]}")
    cfg = _configured().get(key)
    if cfg:
        return _checked(Path(str(cfg)).expanduser(), f"{CONFIG_FILE}:{key}")
    if create:
        default.mkdir(parents=True, exist_ok=True)
    return default


def _checked(path: Path, said_by: str) -> Path:
    if not path.exists():
        raise PathNotConfigured(
            f"{said_by} says {path}, which does not exist. If that is a mount, "
            f"it is not mounted. Refusing rather than falling back to "
            f"~/.neurobrix — a fallback here re-extracts every model onto the "
            f"local disk and calls it normal.")
    if not path.is_dir():
        raise PathNotConfigured(f"{said_by} says {path}, which is not a directory")
    return path


def neurobrix_home() -> Path:
    return _resolve("home", Path.home() / ".neurobrix", create=True)


def cache_dir() -> Path:
    """Extracted models, read at every run."""
    return _resolve("cache", neurobrix_home() / "cache", create=True)


def store_dir() -> Path:
    """`.nbx` containers, as downloaded."""
    return _resolve("store", neurobrix_home() / "store", create=True)


def describe() -> dict:
    """What each location is and WHO said so — for `neurobrix info` and for a
    person who needs to know whether a run read the mount or the local disk."""
    out = {}
    cfg = _configured()
    for key, fn in (("home", neurobrix_home), ("cache", cache_dir), ("store", store_dir)):
        if os.environ.get(_ENV[key]):
            said = f"${_ENV[key]}"
        elif cfg.get(key):
            said = str(CONFIG_FILE)
        else:
            said = "default"
        try:
            out[key] = {"path": str(fn()), "said_by": said}
        except PathNotConfigured as exc:
            out[key] = {"path": None, "said_by": said, "refused": str(exc)}
    return out
