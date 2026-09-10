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


def _entry_numbers() -> list[int]:
    """Every numbered entry, including the ones grouped as a range."""
    found = []
    for line in REGISTER.read_text().splitlines():
        m = re.match(r"^### (\d+)(?:[-–](\d+))?\s+—", line)
        if m:
            first = int(m.group(1))
            last = int(m.group(2)) if m.group(2) else first
            found.extend(range(first, last + 1))
    return found


def test_the_entries_are_contiguous_from_one():
    numbers = _entry_numbers()
    assert numbers, "the register holds no numbered entry"
    assert numbers == list(range(1, len(numbers) + 1)), (
        "the entry numbers are not contiguous from 1. An entry is cited by its "
        "number, from commits and from the other machine's copy, so a number "
        "may never move: strike an entry in place, never delete it.\n"
        f"  found: {numbers}")


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
