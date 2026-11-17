"""The tensor library lives in its own repository; the engine installs its published release and never carries a copy.

The library's code and pages left this repository (owner order 2026-11-17): a copy back at the root would be read
before the installed release by any tool or test that looks for it there, and the engine would again run against a
library no release holds. Red the day `nbxtensor/` or `docs/nbxtensor/` exists again in this repository."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_no_copy_of_the_library_exists_in_the_engine_repository():
    back = [str(p.relative_to(ROOT)) for p in (ROOT / "nbxtensor", ROOT / "docs" / "nbxtensor") if p.exists()]
    assert not back, (f"{back} exist in the engine's repository: the library lives in its own repository "
                      f"(github.com/NeuroBrix/nbxtensor) and the engine installs the release pyproject.toml pins")
