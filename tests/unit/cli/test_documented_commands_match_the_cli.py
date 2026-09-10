"""What the CLI offers and what the docs name are the same list.

A user reported `neurobrix hub` as documented on PyPI and GitHub and absent from
`neurobrix --help`. It does not reproduce: `hub` is registered unconditionally
with a help string, in v0.5.3 and on the trunk, and appears in `--help` here.

But the report pointed at a real hole from the other side. The docs named
fifteen commands; the CLI ships seventeen. **`autotune` and `drift` were
described nowhere.** Two lists, and nothing comparing them.

Run: PYTHONPATH=src python -m pytest tests/unit/cli/test_documented_commands_match_the_cli.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parents[3] / "docs" / "reference" / "cli-commands.md"


def _cli_commands():
    out = subprocess.run([sys.executable, "-m", "neurobrix", "--help"],
                         capture_output=True, text=True).stdout
    return {m.group(1) for m in re.finditer(r"^    ([a-z_]+)\s{2,}\S", out, re.M)}


def _documented():
    return set(re.findall(r"^\| `([a-z_]+)` \|", DOC.read_text(), re.M))


def test_the_reference_page_exists():
    assert DOC.is_file(), f"{DOC} is missing"


def test_every_shipped_command_is_documented():
    missing = sorted(_cli_commands() - _documented())
    assert not missing, (
        f"the engine ships {missing} and no page names them — that is how "
        f"`autotune` and `drift` stayed invisible")


def test_no_documented_command_has_been_removed():
    stale = sorted(_documented() - _cli_commands())
    assert not stale, (
        f"{stale} are documented and the CLI no longer offers them")


def test_the_list_is_not_empty():
    """A gate that compares two empty sets passes and says nothing."""
    assert len(_cli_commands()) > 10, "the CLI help was not parsed"
