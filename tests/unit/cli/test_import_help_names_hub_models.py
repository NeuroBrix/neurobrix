"""`neurobrix import --help` shows models the hub actually serves.

Until 2026-10-12 its examples were `pixart/sigma-xl-1024`,
`deepseek/janus-pro-7b` and `sana/1600m-1024`: slugs the registry never
served, so a newcomer's first copy-paste failed. The examples now come from
one tuple (`cli.utils.IMPORT_EXAMPLES`), held here to the committed record of
the hub catalogue, docs/reference/model-status.md.

Run: PYTHONPATH=src python -m pytest tests/unit/cli/test_import_help_names_hub_models.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from neurobrix.cli.utils import IMPORT_EXAMPLES

STATUS = Path(__file__).resolve().parents[3] / "docs" / "reference" / "model-status.md"
# A model reference shown as an example: after `neurobrix import` or `e.g.,`.
REF = re.compile(r"(?:neurobrix import|e\.g\.,)\s+([\w.-]+/[\w.-]+)")


def _hub_models() -> set[str]:
    """`org/name (registry)` rows whose hub artefact column is not
    "not on the hub"."""
    served = set()
    for line in STATUS.read_text().splitlines():
        m = re.match(r"^\| `[^`]+` \| ([\w.-]+/[\w.-]+) \(registry\) \|", line)
        if m and "| not on the hub |" not in line:
            served.add(m.group(1))
    return served


def _import_help() -> str:
    return subprocess.run([sys.executable, "-m", "neurobrix", "import", "--help"],
                          capture_output=True, text=True, check=True).stdout


def test_the_catalogue_record_is_read():
    """A gate comparing against an empty set passes and says nothing."""
    assert len(_hub_models()) > 10


def test_every_example_is_served_by_the_hub():
    stale = [ref for ref in IMPORT_EXAMPLES if ref not in _hub_models()]
    assert not stale, f"{stale} are not on the hub ({STATUS.name})"


def test_the_help_shows_only_hub_models():
    text = _import_help()
    shown = set(REF.findall(text))
    assert shown, "no model reference found in `import --help`"
    stale = sorted(shown - _hub_models())
    assert not stale, f"`neurobrix import --help` names {stale}, which the hub does not serve"


def test_the_help_points_to_the_live_catalogue():
    assert "neurobrix hub" in _import_help()
