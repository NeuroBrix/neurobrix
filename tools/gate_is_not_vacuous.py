#!/usr/bin/env python3
"""A gate must prove it ran, and a test must prove it could have failed.

This engine has a history of gates that were green because they were empty:

  - the R33 import gate passed on an EMPTY chain of files and nobody saw it for weeks;
  - the upscaler cell asserted the output's dimensions and let a visibly wrong image through;
  - fourteen rows pinned temperature to zero, which meant the sampler code they were meant to
    exercise was never reached;
  - `tests/unit/flow/` and `tests/unit/core/` — thirty tests — failed COLLECTION for months, and
    a directory that cannot be collected reads exactly like a directory that passes.

Every one of those was invisible for the same reason: a suite reports what it ran, never what it
failed to run. Three checks close that, and they are meant to be cheap enough to keep.

    python3 tools/gate_is_not_vacuous.py collect        # each directory collects what its source holds
    python3 tools/gate_is_not_vacuous.py audit          # every test whose assertion cannot fail
    python3 tools/gate_is_not_vacuous.py run            # every test that PASSED without asserting

`collect` and `run` exit non-zero on a finding; `audit` prints a list to triage and exits zero
unless `--strict`.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ASSERTING_CALLS = ("fail", "xfail", "raises", "warns", "deprecated_call", "approx")


def _test_functions(path: Path) -> List[Tuple[str, ast.FunctionDef]]:
    """Every test the source declares: module-level `test_*` and `test_*` methods of `Test*`."""
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError:
        return []
    found: List[Tuple[str, ast.FunctionDef]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            found.append((node.name, node))
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name.startswith("test"):
                    found.append((f"{node.name}::{sub.name}", sub))
    return found


def _module_functions(path: Path) -> Dict[str, ast.AST]:
    """Every function the module defines, so a call to a local helper can be followed."""
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError:
        return {}
    return {n.name: n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _can_fail(fn: ast.AST, helpers: Dict[str, ast.AST] = None, depth: int = 2) -> bool:
    """True when the test holds something that can make it fail on its own terms: an `assert`, a
    `pytest.raises`/`warns` block, a `pytest.fail`, or a unittest `self.assertX`.

    Calls to helpers the same module defines are FOLLOWED, two levels deep. A test that hands its
    checking to `_check(...)` can fail perfectly well, and an audit that reported it would cry
    wolf — which is the failure mode this whole tool exists to close.
    """
    helpers = helpers or {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assert):
            return True
        if isinstance(node, ast.Raise):
            # A test that raises on the path it forbids can fail perfectly well — the stub that
            # says "this wrapper must not be invoked" is a check, not a decoration.
            return True
        if isinstance(node, ast.Attribute) and (node.attr in ASSERTING_CALLS
                                                or node.attr.startswith("assert")):
            return True
        if isinstance(node, ast.Name) and node.id in ("raises", "fail"):
            return True
        if depth > 0 and isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            target = helpers.get(name)
            if target is not None and target is not fn and _can_fail(target, helpers, depth - 1):
                return True
    return False


def _assert_lines(path: Path) -> set:
    """Line numbers of every `assert` statement and `pytest.raises`/`warns` block in a file."""
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError:
        return set()
    lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            lines.add(node.lineno)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                call = item.context_expr
                fn = getattr(call, "func", None)
                name = getattr(fn, "attr", None) or getattr(fn, "id", None)
                if name in ASSERTING_CALLS:
                    lines.add(node.lineno)
    return lines


def _dirs_with_tests(root: Path) -> List[Path]:
    return sorted({p.parent for p in root.rglob("test_*.py")})


def cmd_collect(args) -> int:
    root = Path(args.tests)
    bad = []
    print(f"{'directory':56s} {'source':>7} {'collected':>10}")
    for d in _dirs_with_tests(root):
        declared = sum(len(_test_functions(f)) for f in sorted(d.glob("test_*.py")))
        r = subprocess.run([sys.executable, "-m", "pytest", str(d), "--collect-only", "-q",
                            "-p", "no:cacheprovider", "--no-header"],
                           capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        # `N tests collected, 1 error` is the trap: pytest prints a count AND aborts, so a
        # parser that reads the count alone reports a directory as healthy while none of its
        # tests will run. The exit code and the error line are the signal; the count is not.
        collected, errors = 0, 0
        for line in r.stdout.splitlines():
            t = line.strip()
            if " collected" in t and t[0].isdigit():
                collected = int(t.split()[0])
                if "error" in t:
                    errors = int(t.split("collected,")[1].split()[0])
        if not collected:
            collected = sum(1 for l in r.stdout.splitlines() if "::" in l)
        why = ""
        if r.returncode != 0 or errors:
            why = "collection FAILED — these tests never run"
        elif collected == 0:
            why = "collected nothing"
        elif collected < declared:
            why = "collected fewer than the source declares"
        if why:
            bad.append((str(d), declared, collected, why))
        print(f"{str(d):56s} {declared:7d} {collected:10d}"
              f"{'  <-- ' + why if why else ''}")
    if bad:
        print("\nFINDINGS — a directory that collects less than its source declares is a gate that "
              "cannot fail:")
        for d, declared, collected, why in bad:
            print(f"  {d}: source declares {declared}, pytest collected {collected}. {why}")
        return 1
    print("\nEvery directory collects at least what its source declares.")
    return 0


def cmd_audit(args) -> int:
    root = Path(args.tests)
    findings = []
    total = 0
    for f in sorted(root.rglob("test_*.py")):
        helpers = _module_functions(f)
        for name, fn in _test_functions(f):
            total += 1
            if not _can_fail(fn, helpers):
                findings.append((str(f), name, fn.lineno))
    for path, name, line in findings:
        print(f"{path}:{line}  {name}")
    print(f"\n{len(findings)} of {total} tests carry nothing that can make them fail.")
    return 1 if (findings and args.strict) else 0


def cmd_run(args) -> int:
    """Run the suite under a tracer that records whether each test executed an assertion."""
    plugin = Path(__file__).resolve().parent / "_vacuous_plugin.py"
    out = Path(args.out or "vacuous_report.json")
    env = {**os.environ, "NBX_VACUOUS_REPORT": str(out), "CUDA_VISIBLE_DEVICES":
           os.environ.get("CUDA_VISIBLE_DEVICES", "")}
    r = subprocess.run([sys.executable, "-m", "pytest", args.tests, "-q", "-p", "no:cacheprovider",
                        "-p", f"_vacuous_plugin"], capture_output=True, text=True,
                       env={**env, "PYTHONPATH": f"{plugin.parent}:{env.get('PYTHONPATH','')}"})
    sys.stdout.write(r.stdout[-4000:])
    if not out.exists():
        print("the plugin wrote no report — the run did not reach it", file=sys.stderr)
        return 2
    data = json.loads(out.read_text())
    silent = [t for t, n in data.items() if n == 0]
    print(f"\n{len(silent)} of {len(data)} passing tests executed no assertion.")
    for t in silent[:60]:
        print(f"  {t}")
    return 1 if silent else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("collect", cmd_collect), ("audit", cmd_audit), ("run", cmd_run)):
        p = sub.add_parser(name)
        p.add_argument("--tests", default="tests/unit")
        p.add_argument("--strict", action="store_true")
        p.add_argument("--out", default=None)
        p.set_defaults(func=fn)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
