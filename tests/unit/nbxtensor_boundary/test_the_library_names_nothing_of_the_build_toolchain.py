"""What leaves with the library (nbxtensor/ and docs/nbxtensor/, exported to its own repository) names nothing of the
container build toolchain, which is confidential (supervisor 2026-10-29 17:00: the exported repository holds nothing
of it). Four docstrings named it, one with an internal class, and were carried into the export's history (found
2026-10-29 18:2x by the export's scan, before any push).

What this test would do if the code were wrong: any line naming it (red), whatever its case; "Forget" is a word."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
NAME = re.compile(r"\bforge\b", re.I)


def test_nothing_that_leaves_with_the_library_names_the_build_toolchain():
    hits = []
    for root in (REPO / "nbxtensor", REPO / "docs" / "nbxtensor"):
        for p in root.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                try:
                    text = p.read_text()
                except UnicodeDecodeError:
                    continue
                hits += [f"{p.relative_to(REPO)}:{i}" for i, line in enumerate(text.splitlines(), 1) if NAME.search(line)]
    assert not hits, hits
