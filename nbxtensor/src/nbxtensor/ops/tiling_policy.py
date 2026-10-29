"""The tiling policy's default: the split sizes the tiling rules read (`tiling.yml`, beside this module).

The library owns the file and reads it itself, so the rules run with no application around them. A caller may set
its own table (`set_tiling`, plain data; `None` restores the library's); NeuroBrix's loader hands out this same file
(`core/config/loader.get_tiling_policy`). A missing file is refused by name: no split is ever cut by a size the data
does not state.
"""
from __future__ import annotations

import copy
import functools
from pathlib import Path
from typing import Any, Dict, Optional

_FILE = Path(__file__).resolve().parent / "tiling.yml"

__all__ = ["tiling_table", "set_tiling"]

_SET: Dict[str, Any] = {}          # {"table": dict} once a caller set one


@functools.lru_cache(maxsize=1)
def _raw() -> Dict[str, Any]:
    import yaml
    if not _FILE.is_file():
        raise FileNotFoundError(f"ZERO FALLBACK: the tiling policy is not found: {_FILE}")
    return yaml.safe_load(_FILE.read_text()) or {}


def tiling_table() -> Dict[str, Any]:
    """The policy in force: the table a caller set, else the library's file (a copy: the caller may not edit it)."""
    return copy.deepcopy(_SET["table"] if "table" in _SET else _raw())


def set_tiling(table: Optional[Dict[str, Any]]) -> None:
    """Set the split sizes the tiling rules read (a whole table, as `tiling.yml` writes it); `None` restores the
    library's own."""
    if table is None:
        _SET.clear()
    elif not isinstance(table, dict):
        raise TypeError(f"set_tiling: a table is a dict as tiling.yml writes it, not {type(table).__name__}")
    else:
        _SET["table"] = copy.deepcopy(table)
