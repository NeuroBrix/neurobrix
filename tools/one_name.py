"""The name a tool passes to `neurobrix ... --model`: the model's ONE name.

A tool keys its registers, census rows and verdicts by the CACHE ENTRY (the directory, the container's own
`model_name`), and that key stays. What it hands the engine's command line is the model's one name — the hub name
`neurobrix import` recorded beside the entry (neurobrix/nbx/names.py), which the CLI accepts and for which it refuses
the entry name. The table of names is the catalogue, `docs/internal/catalogue.tsv` (the communication root of
`~/.neurobrix/atelier.json`): its `cache entry` and `hub slug` columns, read by their headers.

    from one_name import cli_name
    cmd = [py, "-m", "neurobrix", "run", "--model", cli_name(entry), ...]

The transition, said exactly:
  * an entry whose record exists       -> its hub slug from the catalogue, which must equal the record (two sources
                                          that disagree are refused by name, never reconciled here);
  * an entry with no record yet        -> the entry name, which the CLI still accepts (installed before 0.6.3);
  * an entry the catalogue does not name, or a name that is not an entry of the cache (a test's fake home, a
    name that is already a slug)       -> returned as it is: the CLI answers it by its own rules.
"""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

_ENTRY_COL = "cache entry"
_SLUG_COL = "hub slug"


class NamesDisagree(RuntimeError):
    """The catalogue and the entry's record give the model two different names."""


def catalogue_path() -> Path:
    import atelier
    return Path(atelier.root("communication")) / "catalogue.tsv"


@lru_cache(maxsize=4)
def table(path: Optional[str] = None) -> Dict[str, str]:
    """cache entry -> hub slug, from the catalogue's header-named columns."""
    p = Path(path) if path else catalogue_path()
    with open(p, newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    header = [h.strip() for h in rows[0]]
    try:
        ei, si = header.index(_ENTRY_COL), header.index(_SLUG_COL)
    except ValueError as e:
        raise RuntimeError(f"ZERO FALLBACK: {p} has no `{_ENTRY_COL}` / `{_SLUG_COL}` column (header: {header})") from e
    return {r[ei].strip(): r[si].strip() for r in rows[1:] if len(r) > max(ei, si) and r[ei].strip()}


def _cache() -> Path:
    """The cache the engine reads (its own resolution: NEUROBRIX_HOME, paths.json, the default)."""
    from neurobrix.core.paths import cache_dir
    return Path(cache_dir())


def cli_name(entry: str, cache: Optional[Path] = None, catalogue: Optional[str] = None) -> str:
    """What `--model` receives for the cache entry `entry` (see the module docstring)."""
    from neurobrix.nbx.names import RECORD    # the record `neurobrix import` writes beside the manifest
    cache = Path(cache) if cache is not None else _cache()
    record_path = cache / entry / RECORD
    if not record_path.is_file():
        return entry
    recorded = json.loads(record_path.read_text())["name"]
    slug = table(catalogue).get(entry)
    if slug is None:
        return recorded                       # an entry outside the catalogue: its record is its one name
    if slug != recorded:
        raise NamesDisagree(f"{entry}: the catalogue names it {slug}, its record {recorded} — fix the one that is "
                            f"wrong (docs/internal/catalogue.tsv or `neurobrix import {slug}`)")
    return slug


def entry_of(name: str, catalogue: Optional[str] = None) -> str:
    """The cache entry a one name designates (the catalogue's inverse); a name that is no hub slug of the catalogue
    is already an entry name and is returned as it is."""
    for entry, slug in table(catalogue).items():
        if slug == name:
            return entry
    return name


def argv_for_tree(argv: list, src) -> list:
    """`argv` (a `neurobrix` command line) as the engine tree at `src` (its `src/` directory) reads it: a tree that
    predates the one name (it has no `neurobrix/nbx/names.py`, so it knows models by their entry only — a base
    commit a gate or a bisection replays) gets the entry back in `--model`; any other tree gets `argv` unchanged."""
    if src is None or (Path(src) / "neurobrix" / "nbx" / "names.py").is_file() or "--model" not in argv:
        return list(argv)
    out = list(argv)
    i = out.index("--model") + 1
    if i < len(out):
        out[i] = entry_of(out[i])
    return out
