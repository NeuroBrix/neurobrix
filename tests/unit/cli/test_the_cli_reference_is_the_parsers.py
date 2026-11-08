"""docs/reference/cli.md is what the `neurobrix` parser says, byte for byte.

The CLI reference was hand-kept: it still said "Version 0.3.0, July 2026" in November, listed flags the parser had
changed and none of the developer commands. It is now written by tools/cli_reference.py from
`neurobrix.cli.create_parser(show_dev=True)`, and this test holds the committed page to the tool's output: a parser
change without a regeneration is red, naming the command that regenerates it.

What would it do if the code were wrong? A stale page (the hand-kept one of 0.3.0) fails `test_the_page_is_the_
parsers` — seen red on main c747d030d before the first regeneration. A rendering that depended on the machine (the
registry URL a help string names comes from NEUROBRIX_REGISTRY) fails `test_the_page_does_not_depend_on_the_machine`.
A rendering that lost a command fails `test_every_command_has_its_section`.

Run: PYTHONPATH=src python -m pytest tests/unit/cli/test_the_cli_reference_is_the_parsers.py
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools" / "cli_reference.py"


def _tool():
    spec = importlib.util.spec_from_file_location("cli_reference", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_page_is_the_parsers():
    tool = _tool()
    committed = tool.DOC.read_text(encoding="utf-8")
    assert committed == tool.render(), (
        f"docs/reference/cli.md differs from the neurobrix parser: run `{tool.REGENERATE}` and commit the page")


def test_the_page_does_not_depend_on_the_machine():
    tool = _tool()
    poisoned = dict(os.environ, NEUROBRIX_REGISTRY="http://registry.machine.invalid:3000")
    text = tool.render(poisoned)
    assert "machine.invalid" not in text, "a machine's NEUROBRIX_REGISTRY reached the reference"
    assert text == tool.render(), "the reference changed with the machine's environment"


def test_every_command_has_its_section():
    tool = _tool()
    text = tool.render()
    import argparse
    from neurobrix.cli import create_parser
    sub = next(a for a in create_parser(show_dev=True)._actions if isinstance(a, argparse._SubParsersAction))
    assert len(sub.choices) > 10, "the parser was not read"
    missing = [n for n in sub.choices if f"\n## {n}\n" not in text]
    assert not missing, f"no section for {missing}"
    for name, child in sub.choices.items():
        for a in child._actions:
            for opt in a.option_strings:
                if opt not in ("-h", "--help"):
                    assert f"`{opt}`" in text, f"{name} {opt} is missing from the reference"
