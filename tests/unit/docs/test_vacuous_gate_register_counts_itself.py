"""The register's stated count is the number of entries it holds.

The register exists because two machines were each counting this class in their
head and each reached "the ninth instance" within an hour of the other. A count
written in prose beside a list is exactly the kind of claim the register itself
is about — so it is pinned here rather than trusted.

Two things are checked, and both are drift, not style:

  * the closing paragraph's number matches the entries actually present;
  * the entry numbers are contiguous from 1, because an entry that is cited by
    number (in a commit, a report, another machine's file) must keep it. A gap
    means someone deleted an entry instead of striking it in place.

Run: PYTHONPATH=src python -m pytest tests/unit/docs/test_vacuous_gate_register_counts_itself.py
"""
from __future__ import annotations

import re
from pathlib import Path

REGISTER = (Path(__file__).resolve().parents[3]
            / "docs" / "reference" / "vacuous-gates-register.md")

_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}


#: Numbers are allocated per MACHINE, agreed between the rack and the Mac on
#: 2026-09-22 after the same number was appended twice in one day for two
#: different defects. Each machine owns a contiguous block and never allocates
#: outside it, so two branches can append without colliding and no citation ever
#: has to move. A block is (first, label) and the list is the whole scheme.
ALLOCATION_BLOCKS = [(1, "the rack"), (500, "the Mac")]


def _entry_numbers() -> list[int]:
    """Every numbered entry, including the ones grouped as a range.

    The heading may carry a parenthetical between the number and the em-dash —
    `### 500 (was 88 — renumbered, see note) — ...` — which is how a renumbered
    entry keeps its old citation readable. The first version of this pattern
    demanded the dash immediately after the number, so those two entries matched
    NOTHING: they were invisible to both cells, the stated count stayed at 93
    while 95 entries were present, contiguity was computed over a list that
    silently excluded them, and the gate was GREEN. A counter that cannot see an
    entry is worse than no counter, and this file is the register's own gate.
    """
    found: list[int] = []
    for line in REGISTER.read_text().splitlines():
        m = re.match(r"^### (\d+)(?:[-–](\d+))?\s*(?:\(([^)]*)\))?\s*[—-]", line)
        if not m:
            continue
        first = int(m.group(1))
        last = int(m.group(2)) if m.group(2) else first
        note = (m.group(3) or "").lower()
        # An ADDENDUM deliberately re-uses its entry's number — it continues that
        # entry rather than adding one — so it is not counted again. Anything else
        # repeating a number is an accident and must be caught, which is why this
        # reads the register's own word instead of silently de-duplicating.
        if "addendum" in note:
            assert first in found, (
                f"entry {first} is marked an addendum but no entry {first} precedes it")
            continue
        for n in range(first, last + 1):
            assert n not in found, (
                f"entry {n} appears twice and the second is not marked an addendum. "
                f"Two defects under one number is exactly what the per-machine "
                f"allocation blocks exist to prevent.")
            found.append(n)
    return found


def _blocks(numbers):
    """`numbers` split into the allocation block each belongs to."""
    out = {first: [] for first, _ in ALLOCATION_BLOCKS}
    starts = sorted(out, reverse=True)
    for n in numbers:
        for first in starts:
            if n >= first:
                out[first].append(n)
                break
    return out


def test_the_entries_are_contiguous_within_each_machines_block():
    """An entry is cited by its number, from commits and from the other machine's
    copy, so a number may never move: strike an entry in place, never delete it.

    Contiguity is per BLOCK. Checked from 1 alone, the Mac's 500 reads as a gap of
    406 missing entries; not checked at all, a deletion inside either block goes
    unnoticed. Both failures are the thing this cell exists to prevent."""
    numbers = _entry_numbers()
    assert numbers, "the register holds no numbered entry"
    for first, label in ALLOCATION_BLOCKS:
        block = sorted(_blocks(numbers)[first])
        if not block:
            continue
        assert block == list(range(first, first + len(block))), (
            f"{label}'s block is not contiguous from {first}.\n  found: {block}")


def test_every_entry_falls_inside_an_allocated_block():
    """A number outside every block is one nobody agreed to own, and it is how the
    collision that created this scheme happened in the first place."""
    lowest = min(first for first, _ in ALLOCATION_BLOCKS)
    for n in _entry_numbers():
        assert n >= lowest, f"entry {n} is below every allocation block"


def test_the_stated_count_matches_the_entries_present():
    text = REGISTER.read_text()
    entries = len(_entry_numbers())
    m = re.search(r"^([A-Za-z]+|\d+) entries,", text, re.M)
    assert m, ("the register no longer states its own count. That sentence is "
               "what makes this a count rather than an impression — restore it.")
    token = m.group(1).lower()
    stated = _WORDS.get(token, int(token) if token.isdigit() else None)
    assert stated == entries, (
        f"the register says {m.group(1)!r} entries and holds {entries}. "
        "Someone appended without updating the closing count — which is the "
        "very defect this file catalogues, committed against the catalogue.")
