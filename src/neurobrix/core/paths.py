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


# ---------------------------------------------------------------------------
# Storage the machine clears, and an installation that depends on it.
#
# Written after the THIRD loss of the Metal backend (2026-09-22). Twice a
# PACKAGE resolved into a session scratchpad; the third time the package was
# durable-looking but the venv it lived in, and the git worktree it was built
# from, were both on `/private/tmp` — and every artefact went with them.
#
# An installation has three legs and losing any one loses the install: the
# package, the environment it is installed into, and the tree it was built
# from. pip records that last one in `direct_url.json`, so it can be read back.
#
# The comparison is by RESOLVED PREFIX, never by substring. `/Users/x/tmpwork`
# is a durable directory; a guard that refused it would be disabled within a
# week, and then it guards nothing.
# ---------------------------------------------------------------------------

def _temp_roots() -> tuple:
    """Every root this machine empties on its own schedule."""
    import tempfile
    named = ["/tmp", "/private/tmp", "/var/tmp", "/private/var/tmp",
             "/var/folders", "/private/var/folders", tempfile.gettempdir()]
    out = []
    for n in named:
        try:
            r = Path(n).resolve()
        except OSError:
            continue
        if r not in out:
            out.append(r)
    return tuple(out)


def ephemeral_reason(path) -> Optional[str]:
    """Why `path` will not survive a reboot or a cleaner, else None."""
    if path is None:
        return None
    p = Path(str(path)).expanduser()
    try:
        rp = p.resolve()
    except OSError:
        rp = p.absolute()
    for root in _temp_roots():
        if rp == root or root in rp.parents:
            return f"under {root}, which the machine clears without asking"
    return None


def _dist_info_dirs(module_name: str, search_root: Optional[Path]):
    roots = [search_root] if search_root is not None else []
    if search_root is None:
        import sys as _sys
        seen = set()
        for entry in _sys.path:
            if not entry or entry in seen:
                continue
            seen.add(entry)
            roots.append(Path(entry))
    for root in roots:
        try:
            yield from Path(root).glob(f"{module_name.replace('-', '_')}-*.dist-info")
            yield from Path(root).glob(f"{module_name.replace('_', '-')}-*.dist-info")
        except OSError:
            continue


def installation_refusals(module_name: str, search_root=None) -> list:
    """Every ephemeral leg `module_name`'s installation stands on.

    Empty list means all three legs are durable. Each entry is a sentence naming
    WHICH leg and WHERE, because "refused" without the place is unactionable.
    """
    import sys as _sys
    reasons: list = []

    # 1. the package itself
    pkg_dir = None
    if search_root is not None:
        cand = Path(search_root) / module_name
        pkg_dir = cand if cand.exists() else None
    else:
        try:
            import importlib.util
            spec = importlib.util.find_spec(module_name)
            if spec is not None and spec.origin:
                pkg_dir = Path(spec.origin).parent
        except (ImportError, ValueError):
            pkg_dir = None
    if pkg_dir is not None:
        why = ephemeral_reason(pkg_dir)
        if why:
            reasons.append(f"the package {module_name} is installed at {pkg_dir}, {why}")

    # 2. the environment it is installed into
    why = ephemeral_reason(_sys.prefix)
    if why:
        reasons.append(f"the environment at {_sys.prefix} is {why}")

    # 3. the tree it was built from, as pip recorded it
    for dist in _dist_info_dirs(module_name, search_root):
        record = dist / "direct_url.json"
        if not record.exists():
            continue
        try:
            url = json.loads(record.read_text()).get("url") or ""
        except (OSError, ValueError):
            continue
        if not url.startswith("file://"):
            continue
        tree = url[len("file://"):]
        why = ephemeral_reason(tree)
        if why:
            reasons.append(f"{module_name} was built from {tree}, {why}")
    return reasons
