#!/usr/bin/env python3
"""Write docs/reference/cli.md from the `neurobrix` argument parser — the CLI reference is generated, never hand-kept.

Every subcommand (the developer ones that `--help` hides included, marked as such), every nested action
(`autotune certify`, ...), every flag with its value, its default and its help, the description and the epilog
examples: all read from `neurobrix.cli.create_parser(show_dev=True)`. Nothing is typed here about a command.

Contract (tests/unit/cli/test_the_cli_reference_is_the_parsers.py):
  * the committed docs/reference/cli.md is byte-identical to `render()`; a parser change without a regeneration is
    red, naming the command to run;
  * the text does not depend on the machine: the parser is read in a child process whose environment carries no
    NEUROBRIX_* / NBX_* variable (the registry URL a help string names is the public default, never this machine's
    registry) and whose terminal width is fixed;
  * the output is never empty: every subcommand of the parser has its section, or the tool refuses by name.

Usage:
    PYTHONPATH=src python tools/cli_reference.py            # write docs/reference/cli.md
    PYTHONPATH=src python tools/cli_reference.py --check    # exit 1 when the committed file differs
    PYTHONPATH=src python tools/cli_reference.py --stdout   # print the text (what the child process runs)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "reference" / "cli.md"
REGENERATE = "PYTHONPATH=src python tools/cli_reference.py"

#: Environment prefixes the parser's help strings may read (the registry URL, the engine home): removed so the
#: reference is the same on every machine.
_MACHINE_PREFIXES = ("NEUROBRIX_", "NBX_")


def _cell(text) -> str:
    """One markdown line: whitespace folded, pipes and angle brackets escaped."""
    return " ".join(str(text).split()).replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;")


def _value_name(action: argparse.Action) -> str:
    if action.nargs == 0:
        return ""
    if action.metavar is not None:
        return str(action.metavar) if not isinstance(action.metavar, tuple) else " ".join(action.metavar)
    if action.choices is not None:
        return "{" + ",".join(str(c) for c in action.choices) + "}"
    return action.dest.upper()


def _default(action: argparse.Action) -> str:
    if action.required:
        return "required"
    d = action.default
    if d is None or d is argparse.SUPPRESS:
        return "—"
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return "off"
    if isinstance(action, argparse._CountAction):
        return str(d)
    return f"`{d}`"


def _subparsers(parser: argparse.ArgumentParser):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def _synopsis(path: str, parser: argparse.ArgumentParser) -> str:
    words = [path]
    if any(a.option_strings for a in parser._actions if not isinstance(a, argparse._HelpAction)):
        words.append("[options]")
    for a in parser._actions:
        if a.option_strings or isinstance(a, argparse._SubParsersAction):
            continue
        name = a.metavar or a.dest
        words.append(f"{name} ..." if a.nargs in ("+", "*") else (f"[{name}]" if a.nargs == "?" else name))
    sub = _subparsers(parser)
    if sub is not None:
        words.append("{" + ",".join(sub.choices) + "}")
    return " ".join(words)


def _section(path: str, parser: argparse.ArgumentParser, help_line: str, dev: bool, level: int) -> list:
    out = [f"{'#' * level} {path.split(' ', 1)[1] if ' ' in path else path}", ""]
    if dev:
        out += ["*Developer command: listed by `neurobrix --help -v`, not by `neurobrix --help`.*", ""]
    if help_line:
        out += [_cell(help_line) + ("" if help_line.rstrip().endswith(".") else "."), ""]
    desc = (parser.description or "").strip("\n").rstrip()
    if desc and _cell(desc) != _cell(help_line):
        # a description laid out on several lines (a list of levels) keeps its layout; a sentence is a paragraph
        out += (["```text", desc, "```", ""] if "\n" in desc.strip() else [_cell(desc), ""])
    out += ["```", _synopsis(path, parser), "```", ""]
    rows = []
    for a in parser._actions:
        if isinstance(a, (argparse._HelpAction, argparse._SubParsersAction)):
            continue
        if a.option_strings:
            flag = ", ".join(f"`{o}`" for o in a.option_strings)
            value = _value_name(a)
            if value:
                flag += f" `{value}`"
        else:
            flag = f"`{a.metavar or a.dest}`"
        rows.append(f"| {flag} | {_default(a)} | {_cell(a.help or '')} |")
    if rows:
        out += ["| argument | default | description |", "|---|---|---|", *rows, ""]
    if parser.epilog and parser.epilog.strip():
        out += ["```text", parser.epilog.strip("\n").rstrip(), "```", ""]
    sub = _subparsers(parser)
    if sub is not None:
        helps = {a.dest: a.help for a in sub._choices_actions}
        for name, child in sub.choices.items():
            out += _section(f"{path} {name}", child, helps.get(name) or "", dev, level + 1)
    return out


def render_here() -> str:
    """The reference, from the parser of the neurobrix this process imports."""
    from neurobrix.cli import DEV_COMMANDS, create_parser

    parser = create_parser(show_dev=True)
    sub = _subparsers(parser)
    if sub is None or not sub.choices:
        raise SystemExit("cli_reference: the neurobrix parser has no subcommands — nothing to document")
    helps = {a.dest: a.help for a in sub._choices_actions}
    user = [n for n in sub.choices if n not in DEV_COMMANDS]
    dev = [n for n in sub.choices if n in DEV_COMMANDS]

    out = [
        "# NeuroBrix CLI reference",
        "",
        f"<!-- Generated by tools/cli_reference.py from the `neurobrix` argument parser. Do not edit by hand: "
        f"regenerate with `{REGENERATE}`. -->",
        "",
        "Every command, flag, default and help text below is read from the command-line parser itself. How the "
        "commands are used, family by family, is in [CLI usage](../cli_usage.md); serving, chat and the HTTP API "
        "in [Serving and chat](../serving.md).",
        "",
        "A default of `—` means the value comes from elsewhere: the model's container, its family's defaults, "
        "the hardware profile, or the command's own behaviour described in the flag's text.",
        "",
        "## Commands",
        "",
        "| command | description |",
        "|---|---|",
        *[f"| [`{n}`](#{n}) | {_cell(helps.get(n) or '')} |" for n in user],
        "",
        "Developer commands (`neurobrix --help -v`):",
        "",
        "| command | description |",
        "|---|---|",
        *[f"| [`{n}`](#{n}) | {_cell(helps.get(n) or '')} |" for n in dev],
        "",
        "## Global options",
        "",
        "```",
        "neurobrix [options] <command> ...",
        "```",
        "",
        "| argument | default | description |",
        "|---|---|---|",
    ]
    for a in parser._actions:
        if isinstance(a, (argparse._HelpAction, argparse._SubParsersAction)):
            continue
        out.append(f"| {', '.join(f'`{o}`' for o in a.option_strings)} | {_default(a)} | {_cell(a.help or '')} |")
    out.append("")
    for n in user + dev:
        out += _section(f"neurobrix {n}", sub.choices[n], helps.get(n) or "", n in DEV_COMMANDS, 2)
    text = "\n".join(out).rstrip("\n") + "\n"
    missing = [n for n in sub.choices if f"\n## {n}\n" not in text]
    if missing:
        raise SystemExit(f"cli_reference: no section written for {missing}")
    return text


def machine_free_env(base=None) -> dict:
    env = {k: v for k, v in (os.environ if base is None else base).items() if not k.startswith(_MACHINE_PREFIXES)}
    env.update({"COLUMNS": "120", "NO_COLOR": "1", "PYTHON_COLORS": "0"})
    src = str(REPO / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def render(base_env=None) -> str:
    """The reference, read in a child process with no machine-specific variable (see the module contract)."""
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--stdout"], cwd=str(REPO),
                          env=machine_free_env(base_env), capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(f"cli_reference: rendering failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}")
    if not proc.stdout.strip():
        raise SystemExit("cli_reference: the rendering printed nothing")
    return proc.stdout


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--stdout", action="store_true", help="print the reference (this process's parser)")
    g.add_argument("--check", action="store_true", help="exit 1 when docs/reference/cli.md differs")
    args = ap.parse_args(argv)
    if args.stdout:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdout.write(render_here())
        return 0
    text = render()
    if args.check:
        current = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
        if current != text:
            print(f"{DOC.relative_to(REPO)} differs from the parser: run `{REGENERATE}`", file=sys.stderr)
            return 1
        print(f"{DOC.relative_to(REPO)} matches the parser")
        return 0
    DOC.write_text(text, encoding="utf-8")
    print(f"wrote {DOC.relative_to(REPO)} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
