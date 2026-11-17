"""NBXTensor is ONE library, and the engine's scatter of tensor semantics can only shrink.

The rule: NBXTensor is NeuroBrix's tensor library, a library of its own that the engine imports. `tools/scatter_scan.py` finds the tensor-semantics definitions still written in the engine;
`tools/scatter_list.tsv` is the engine's reviewed list of them, each with its destination in the library.

1. A definition found in the engine and NOT on the list fails: the scatter grew. Write it in nbxtensor instead.
2. A list entry the scan no longer finds fails: it moved. Delete its line.
3. An entry without its destination fails.
4. The count never exceeds the list's header, and the header states it truthfully.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("scatter_scan", REPO / "tools" / "scatter_scan.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def _scan():
    return {(r[0], r[1]) for r in S.scan()}


def test_every_scattered_definition_is_on_the_engines_list():
    grew = sorted(_scan() - set(S.read_list()[1]))
    assert not grew, ("Tensor semantics written in the engine instead of nbxtensor (write it in the library, or "
                      "justify it in scatter_scan.NOT_SEMANTIC): " + "; ".join(f"{p}:{s}" for p, s in grew))


def test_the_list_names_nothing_that_already_moved():
    moved = sorted(set(S.read_list()[1]) - _scan())
    assert not moved, (f"These entries moved into nbxtensor: delete their lines from {S.LIST.name}: "
                       + "; ".join(f"{p}:{s}" for p, s in moved))


def test_every_entry_names_its_destination_in_the_library():
    blank = sorted(k for k, d in S.read_list()[1].items() if not d.strip())
    assert not blank, f"entries without a destination: {blank}"


def test_the_scatter_only_goes_down():
    head, rows = S.read_list()
    assert "definitions" in head, f"{S.LIST.name}: its header states no definitions count"
    assert len(rows) == int(head["definitions"]), f"the list holds {len(rows)}, its header says {head['definitions']}"
    assert len(_scan()) <= int(head["definitions"]), (
        f"the engine holds {len(_scan())} scattered definitions (the header: {head['definitions']}): it only shrinks")
