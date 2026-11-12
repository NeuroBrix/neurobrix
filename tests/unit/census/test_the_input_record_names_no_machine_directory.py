"""The census input record names every file it read by a root it knows (`repo:`, `cache:`, `home:`, `comm:`), never by a
machine's directory: the catalogue a derivation reads lives in the workshop's communication directory, outside a worktree,
and was committed as an absolute path in both volta *.inputs.jsonl (qa, 2026-11-12).

What this test does if the code is wrong: it fails when the catalogue's path records as an absolute path, or does not read
back to the same file (seen red on main ac107bba4: no `comm:` root)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import census_inputs as C   # noqa: E402


def test_the_catalogue_records_under_the_communication_root_and_reads_back():
    catalogue = C._communication_root() / "catalogue.tsv"
    name = C._rel(catalogue)
    assert name == "comm:catalogue.tsv", name
    assert C._abs(name) == catalogue.resolve()
