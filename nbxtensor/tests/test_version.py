"""The library carries one version number: `nbxtensor/pyproject.toml` and `nbxtensor.__version__` state the same, and
it follows the numbering of docs/nbxtensor/DESIGN.md section 7 — plain MAJOR.MINOR.PATCH, no pre-release label.

What this test would do if the code were wrong: a version bumped in one place only (red), a label such as "0.0.1a1"
or "0.0.1.dev0" (red)."""
from __future__ import annotations

import re
from pathlib import Path

import nbxtensor

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_one_version_number():
    declared = re.search(r'^version\s*=\s*"([^"]+)"', PYPROJECT.read_text(), re.M).group(1)
    assert declared == nbxtensor.__version__, (declared, nbxtensor.__version__)


def test_the_version_carries_no_pre_release_label():
    assert re.fullmatch(r"\d+\.\d+\.\d+", nbxtensor.__version__), nbxtensor.__version__
    assert nbxtensor.__version__ != "0.0.0", "the first publication is 0.0.1"
