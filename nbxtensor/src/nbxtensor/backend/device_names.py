"""The arch string a device is known by: the key its device file and its certified kernel settings are found under.

A device reports a model name ("Apple M4 Pro", "Tesla V100-SXM2-16GB"); the arch string is that name lowercased with
spaces hyphenated — "apple-m4-pro". One rule, owned by the library (it was a service the application provided,
`nbxtensor.host`), read by the Metal backend and by any application keying its own files the same way.
"""
from __future__ import annotations

import functools
from typing import Optional

__all__ = ["arch_string"]


@functools.lru_cache(maxsize=16)
def arch_string(model: Optional[str]) -> str:
    """The arch string of a device's model name: lowercased, spaces hyphenated ("Apple M4 Pro" -> "apple-m4-pro")."""
    return str(model or "").strip().lower().replace(" ", "-")
