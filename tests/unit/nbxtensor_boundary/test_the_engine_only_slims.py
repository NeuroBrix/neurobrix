"""The engine only slims (Hocine 2026-10-29 22:39): its Python lines and its open slimming findings may only go down.

`tools/engine_slim_scan.py` lists what the engine carries that nbxtensor makes unnecessary; every finding is either
open in `docs/nbxtensor/engine_slim_list.tsv` or kept, with its reason, in `docs/nbxtensor/ENGINE_SLIM.md`. The same
two rules as the scatter guard, so the debt decreases by construction, and a ratchet on the two figures:

1. A finding neither open nor kept fails: something was written into the engine that the library makes unnecessary
   (delete it, move it into nbxtensor, or write why it stays).
2. An open or kept finding the scan no longer finds fails: it was resolved — delete its line.
3. The engine's line count and the number of open findings never exceed the list's header; the header states the
   open count truthfully.

What would this file do if the engine grew? A new dead or duplicated definition: rule 1 RED; new lines: rule 3 RED."""
from __future__ import annotations

import functools
import importlib.util
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LIST = REPO / "docs" / "nbxtensor" / "engine_slim_list.tsv"
DOC = REPO / "docs" / "nbxtensor" / "ENGINE_SLIM.md"
_spec = importlib.util.spec_from_file_location("engine_slim_scan", REPO / "tools" / "engine_slim_scan.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


@functools.lru_cache(maxsize=1)
def _found():
    return {(k, p, s) for k, p, s, _ in S.scan()}


def _header(name: str) -> int:
    m = re.search(rf"^# {name}: (\d+)$", LIST.read_text(), re.M)
    assert m, f"{LIST}: its header states no {name}"
    return int(m.group(1))


def _open():
    rows = [line.split("\t") for line in LIST.read_text().splitlines()
            if line.strip() and not line.startswith("#") and not line.startswith("kind\t")]
    return {(r[0], r[1], r[2]) for r in rows}


def _kept():
    """The `Kept in the engine` table of ENGINE_SLIM.md: (kind, path, symbol) with a reason."""
    text = DOC.read_text()
    section = text.split("## Kept in the engine", 1)[1]
    out = set()
    for line in section.splitlines():
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[0] in ("dead", "duplicate", "wrapper", "fact"):
            assert cells[3], f"kept without a reason: {line}"
            out.add((cells[0], cells[1], cells[2]))
    return out


def test_every_finding_is_open_or_kept_with_its_reason():
    new = sorted(_found() - _open() - _kept())
    assert not new, ("Written into the engine though nbxtensor makes it unnecessary — delete it, move it into the "
                     "library, or write why it stays in docs/nbxtensor/ENGINE_SLIM.md: "
                     + "; ".join(f"{k} {p}:{s}" for k, p, s in new[:40]) + (" ..." if len(new) > 40 else ""))


def test_nothing_listed_is_already_resolved():
    gone = sorted((_open() | _kept()) - _found())
    assert not gone, ("Resolved — delete their lines from docs/nbxtensor/engine_slim_list.tsv or ENGINE_SLIM.md: "
                      + "; ".join(f"{k} {p}:{s}" for k, p, s in gone[:40]))


def test_the_engine_and_its_open_findings_only_go_down():
    lines, opened = S.engine_lines(), len(_open())
    assert opened == _header("open"), f"the list holds {opened} open findings, its header says {_header('open')}"
    assert lines <= _header("engine_lines"), (
        f"the engine grew to {lines} Python lines (the list's header: {_header('engine_lines')}): it only slims")
