"""The autotune's two settings an application may set, in a module that loads nothing but the standard library — so
an application can set them when it starts without importing the autotune itself.

- `set_store(path)`: the directory of proven kernel settings (a path, or a callable returning one) (None: the library's own, the user's cache,
  `backend.device_facts.certified_dir`); `directory()` the one in force.
- `set_miss_explainer(fn)`: `fn(kernel_qual, key, memory_gb) -> str`, an application's account of a key a confirmation
  run misses (where it expected that key to come from), appended to the refusal; None removes it.

`ops.autotune_certified` reads both at call time.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Optional

__all__ = ["set_store", "directory", "set_miss_explainer", "miss_explainer"]

_STORE: Dict[str, object] = {}        # {"dir": a Path, or a callable returning one} once a caller set its store
_MISS_EXPLAINER: Optional[Callable] = None


def set_store(path) -> None:
    """Set the directory of proven kernel settings: a path, or a callable returning one (read each time the store is
    asked — an application can set it when it starts without resolving it then); `None` restores the library's own,
    the user's cache (`backend.device_facts.certified_dir`)."""
    if path is None:
        _STORE.clear()
    else:
        _STORE["dir"] = path if callable(path) else Path(path)


def directory() -> Path:
    """The store in force: NEUROBRIX_AUTOTUNE_CERTIFIED_DIR when set (tests, a contributor's draft), else the one a
    caller set (`set_store`), else the library's own in the user's cache."""
    override = os.environ.get("NEUROBRIX_AUTOTUNE_CERTIFIED_DIR")
    if override:
        return Path(override)
    if "dir" in _STORE:
        where = _STORE["dir"]
        return Path(where()) if callable(where) else where
    from nbxtensor.backend.device_facts import certified_dir
    return certified_dir()


def set_miss_explainer(explain) -> None:
    """Install `explain(kernel_qual, key, memory_gb) -> str`: where the application expected a key it finds uncertified
    to come from, appended to the refusal; None removes it. Optional: without one, the refusal says what the store holds
    and nothing more."""
    global _MISS_EXPLAINER
    if explain is not None and not callable(explain):
        raise TypeError("set_miss_explainer: the explainer must be callable (kernel_qual, key, memory_gb) -> str")
    _MISS_EXPLAINER = explain


def miss_explainer() -> Optional[Callable]:
    """The explainer in force, or None."""
    return _MISS_EXPLAINER
