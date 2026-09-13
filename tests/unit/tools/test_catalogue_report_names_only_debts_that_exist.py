"""The catalogue document names, beside each line, the debts that hold it. A
name that is not a heading of DETTE.md is a defect of the table, not of the
debt file — and DETTE.md is gitignored on this rig, so the test skips where
the file is absent rather than reading a plausible reconstruction."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def test_every_named_debt_is_a_heading_of_the_debt_file():
    dette = REPO / "DETTE.md"
    if not dette.exists():
        pytest.skip("DETTE.md is not on this checkout")
    sys.path.insert(0, str(REPO / "tools"))
    import catalogue_state_report as R
    text = dette.read_text(encoding="utf-8")
    named = [n.split(" ")[0] for names in list(R.DEBTS_BY_CONTAINER.values()) + list(R.DEBTS_BY_SLUG.values())
             for n in names]
    absent = [n for n in named if n not in text]
    assert not absent, f"named in the report, not in DETTE.md: {absent}"
    assert R.debts_cell("mochi-1-preview", "Mochi-1-preview").startswith("`D-MOCHI")
    assert R.debts_cell("TinyLlama-1.1B-Chat", "TinyLlama-1.1B-Chat") == "none named"
