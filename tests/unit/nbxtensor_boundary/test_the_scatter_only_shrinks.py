"""NBXTensor is ONE library, and the engine's scatter of tensor semantics can only shrink.

Hocine's rule (28/10/2026): NBXTensor is NeuroBrix's master piece, a library of
its own that the engine imports. On 28/10 it was not: the dtype size had four
readers, bf16 conversions lived in three files, a second allocator lived in
replay.py, the view rule lived in the dispatcher, and Prism re-derived the
allocator's behaviour with coefficients. `nbxtensor/tools/scatter_scan.py`
finds those definitions; `docs/nbxtensor/migration_list.tsv` is their reviewed
list, each with its destination in the library.

Two rules, the same as the R33 guard's, so the debt decreases by construction:

1. A tensor-semantics definition found in the engine and NOT on the list fails:
   the scatter grew. Write it in nbxtensor instead.
2. A list entry the scan no longer finds fails: it moved. Delete its line.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LIST = REPO / "docs" / "nbxtensor" / "migration_list.tsv"
SCAN = REPO / "nbxtensor" / "tools" / "scatter_scan.py"


def _scan():
    spec = importlib.util.spec_from_file_location("scatter_scan", SCAN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {(r[0], r[1]) for r in mod.scan()}


def _listed():
    out = {}
    for line in LIST.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        path, symbol, dest = line.split("\t")[:3]
        out[(path, symbol)] = dest
    return out


def test_every_scattered_definition_is_on_the_migration_list():
    grew = sorted(_scan() - set(_listed()))
    assert not grew, (
        "Tensor semantics written in the engine instead of nbxtensor "
        "(write it in the library, or justify it in scatter_scan.NOT_SEMANTIC): "
        + "; ".join(f"{p}:{s}" for p, s in grew))


def test_the_migration_list_names_nothing_that_already_moved():
    moved = sorted(set(_listed()) - _scan())
    assert not moved, (
        "These entries moved into nbxtensor — delete their lines from "
        "docs/nbxtensor/migration_list.tsv: "
        + "; ".join(f"{p}:{s}" for p, s in moved))


def test_every_entry_names_its_destination_in_the_library():
    blank = sorted(k for k, d in _listed().items() if not d.strip())
    assert not blank, f"entries without a destination: {blank}"
