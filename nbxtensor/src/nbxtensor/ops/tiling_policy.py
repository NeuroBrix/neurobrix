"""The tiling policy's default: the split sizes the tiling rules read (`tiling.yml`, beside this module).

The library owns the file and reads it itself, so the rules run with no application around them; an application
may provide its own policy through `nbxtensor.host` (`tiling_policy`), and NeuroBrix's loader hands out this same
file (`core/config/loader.get_tiling_policy`). A missing file is refused by name: no split is ever cut by a size the
data does not state.
"""
from __future__ import annotations

import copy
import functools
from pathlib import Path
from typing import Any, Dict

_FILE = Path(__file__).resolve().parent / "tiling.yml"

__all__ = ["tiling_table"]


@functools.lru_cache(maxsize=1)
def _raw() -> Dict[str, Any]:
    import yaml
    if not _FILE.is_file():
        raise FileNotFoundError(f"ZERO FALLBACK: the tiling policy is not found: {_FILE}")
    return yaml.safe_load(_FILE.read_text()) or {}


def tiling_table() -> Dict[str, Any]:
    """The whole policy as written (a copy: the caller may not edit the library's)."""
    return copy.deepcopy(_raw())
