"""A command recorded with `" ".join` is not the command that ran.

Raised by the Mac on 2026-09-23. Their census recorded `" ".join(argv)`, so a multi-word
`--prompt` lost its quoting and split at every space. Re-parsed, `--prompt "a red apple rolling
slowly across a wooden table"` becomes `--prompt a` followed by eight stray positionals.

**What it cost on their side**: three of ten verification cells had never run, because the
recorded command could not be replayed. Once it could, those three formed **73 keys the
3 106-key census had never named** — 65 from chatterbox, 8 from Kokoro. The record being
unreplayable had silently removed a third of a verification battery from service.

**Measured here the same day, and the rack had it in three places**:

    tools/certified_census.py:193        "command": " ".join(cmd[2:])
    tools/precision_zoo_campaign.py:209  fh.write("$ " + " ".join(cmd))
    src/neurobrix/cli/commands/drift.py:66  fh.write("$ " + " ".join(cmd))

A record of what ran is evidence, and evidence that cannot be replayed is a note. `shlex.join`
is the whole fix and has been in the stdlib since 3.8; `tools/flightrec.py` already used it,
which is what made the other three findable by contrast rather than by luck.
"""
from __future__ import annotations

import ast
import shlex
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

#: Every file that turns an argv list back into a string. A new one joining with `" "` is the
#: same defect again, so the check is on the TREE, not on a list of known sites.
SEARCHED = ("tools", "src/neurobrix")


def _joins_argv_unsafely(path: Path):
    """(line, source) for every `" ".join(x)` whose argument looks like a command list."""
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError):
        return []
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "join"):
            continue
        sep = node.func.value
        if not (isinstance(sep, ast.Constant) and sep.value == " "):
            continue
        if not node.args:
            continue
        arg = node.args[0]
        name = (arg.id if isinstance(arg, ast.Name) else
                arg.value.id if isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name)
                else getattr(getattr(arg, "attr", None), "__str__", lambda: "")())
        if str(name).lower() in {"cmd", "argv", "command", "args", "sys.argv"}:
            out.append((node.lineno, str(name)))
    return out


# ───────────────────────── the invariant, over the tree ─────────────────────────

def test_no_file_joins_a_command_list_with_a_bare_space():
    offenders = {}
    for sub in SEARCHED:
        for p in (ROOT / sub).rglob("*.py"):
            if "test_" in p.name:
                continue
            bad = _joins_argv_unsafely(p)
            if bad:
                offenders[str(p.relative_to(ROOT))] = bad
    assert not offenders, (
        "a command list is joined with a bare space, so the record cannot be replayed: a "
        "multi-word argument splits at every space. Use `shlex.join`.\n"
        + "\n".join(f"  {f}: line(s) {[l for l, _ in v]}" for f, v in offenders.items()))


# ───────────────────── what the defect actually does, as arithmetic ─────────────────────

def test_a_bare_space_join_destroys_a_multi_word_argument():
    """The defect itself, so the cell above is not asserting a style preference."""
    cmd = ["neurobrix", "run", "--model", "mochi-1-preview",
           "--prompt", "a red apple rolling slowly across a wooden table", "--steps", "4"]
    broken = shlex.split(" ".join(cmd))
    assert broken[broken.index("--prompt") + 1] == "a"
    assert len(broken) > len(cmd), "the record gained tokens that were never arguments"


def test_shlex_join_round_trips_exactly():
    """And the fix, as the same arithmetic."""
    cmd = ["neurobrix", "run", "--model", "mochi-1-preview",
           "--prompt", "a red apple rolling slowly across a wooden table", "--steps", "4"]
    assert shlex.split(shlex.join(cmd)) == cmd


@pytest.mark.parametrize("arg", [
    "a red apple rolling slowly across a wooden table",
    "quotes 'inside' the prompt",
    'double "quotes" too',
    "a trailing space ",
    "semicolon; and $DOLLAR and `backtick`",
])
def test_shlex_join_round_trips_the_awkward_cases_too(arg):
    """A prompt is user text and carries whatever a user typed."""
    cmd = ["neurobrix", "run", "--prompt", arg]
    assert shlex.split(shlex.join(cmd)) == cmd


# ───────────────────── the sites the Mac's finding named ─────────────────────

@pytest.mark.parametrize("rel", [
    "tools/certified_census.py",
    "tools/precision_zoo_campaign.py",
    "src/neurobrix/cli/commands/drift.py",
    "tools/flightrec.py",
])
def test_the_known_recorders_use_shlex(rel):
    """Named individually as well as caught by the tree scan, because these four are the ones
    whose records are read back as evidence. `flightrec.py` is included because it was ALREADY
    correct and is what made the other three findable by contrast."""
    src = (ROOT / rel).read_text()
    assert "shlex.join" in src, f"{rel} records a command without shlex.join"
