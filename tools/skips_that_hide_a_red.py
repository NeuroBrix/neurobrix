#!/usr/bin/env python3
"""Which tests would SKIP where they should FAIL.

Written 2026-09-12 after committing the pattern in the file written to enforce
the rule against it. A module-level

    try:
        from x import a, b
        HAS = True
    except Exception:
        HAS = False

    requires = pytest.mark.skipif(not HAS, reason="x needed")

turns every reason `x` cannot be imported into the same skip. When `a` exists
and `b` does not -- the ordinary state of a test written before its code --
five red tests became five skips, and a skip is invisible in a count.

The pattern is not wrong in itself. A test that needs Metal on a machine
without Metal must skip. What makes it dangerous is SCOPE: the guard covers a
whole import list, so it swallows the absence of the very function under test.

Two questions, and only the pair is useful:

  --scan     every module-level try/except around imports that feeds a skip,
             with what it swallows. An upper bound: a guard over a single
             third-party package is usually right.

  --reasons  every `skip`/`skipif` whose reason is empty or generic. A skip
             without a reason cannot be audited later, and a suite with three
             hundred of them will never be asked why.

    tools/skips_that_hide_a_red.py --scan [--verbose]
    tools/skips_that_hide_a_red.py --reasons
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"

def _swallowed_names(node: ast.Try) -> list[str]:
    """The NAMES a module-level catch-all would swallow.

    The distinction that matters is not which package is being imported -- a
    first version keyed on a hand-written list of "environment" packages,
    which is arbitrary and puts our own modules on the wrong side. It is
    whether the guard imports a MODULE or NAMES OUT OF one.

    `import x` fails only when `x` is unimportable, which is the environment
    fact a skip is for. `from x import a, b` fails identically when `a` is
    gone -- and `a` is usually the function under test, so its absence reads
    as the absence of the machine.
    """
    out = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.ImportFrom):
            for a in sub.names:
                out.append(f"{sub.module or ''}.{a.name}")
    return out


def _catches_everything(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    name = ast.unparse(handler.type)
    return name in ("Exception", "BaseException") or "Exception" in name


def scan(verbose: bool) -> int:
    total_files = 0
    risky = []
    for path in sorted(TESTS.rglob("test_*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        total_files += 1
        for node in tree.body:                      # MODULE level only
            if not isinstance(node, ast.Try):
                continue
            if not any(_catches_everything(h) for h in node.handlers):
                continue
            swallowed = _swallowed_names(node)
            if swallowed:
                risky.append((path.relative_to(ROOT), node.lineno, swallowed))

    print(f"test files scanned                        : {total_files}")
    print(f"with a module-level catch-all around imports: {len(risky)}")
    print()
    if risky:
        for rel, lineno, swallowed in risky:
            print(f"  {rel}:{lineno}")
            print(f"      swallows: {', '.join(swallowed[:6])}"
                  + (" ..." if len(swallowed) > 6 else ""))
    print()
    print("A guard around `import x` is the legitimate form and is not listed:")
    print("it fails only when x is unimportable, which is what a skip is for.")
    print("What IS listed guards `from x import a` -- so the absence of `a`,")
    print("usually the function under test, reads as the absence of the")
    print("machine, and a red becomes a skip.")
    print()
    print("The fix is scope, not removal: guard the package import, and import")
    print("the names inside the tests, where a missing one fails.")
    return 0


def reasons() -> int:
    empty = []
    total = 0
    for path in sorted(TESTS.rglob("test_*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = ast.unparse(node.func)
            if not (fn.endswith("skip") or fn.endswith("skipif")):
                continue
            total += 1
            # A reason is ABSENT only when no reason argument is given, or it
            # is an empty literal. The first version of this check accepted
            # only `ast.Constant` and so reported an f-string reason and a
            # `reason=x or "..."` as missing -- it counted 84 of 162 where the
            # true number is far smaller. A detector that reads the node TYPE
            # instead of asking whether a reason is there is the same error as
            # reading where a value is named: it answers about the form, not
            # about the fact.
            node_reason = None
            for kw in node.keywords:
                if kw.arg == "reason":
                    node_reason = kw.value
            if node_reason is None and node.args:
                node_reason = node.args[-1]
            if node_reason is None:
                empty.append((path.relative_to(ROOT), node.lineno, "(no reason argument)"))
                continue
            if isinstance(node_reason, ast.Constant):
                text = str(node_reason.value or "")
                if len(text.strip()) < 4:
                    empty.append((path.relative_to(ROOT), node.lineno, repr(text)))
                continue
            # Any expression -- f-string, `or`, a variable -- IS a reason. It
            # cannot be read statically and is not claimed to be missing.
    print(f"skip / skipif sites          : {total}")
    print(f"without a usable reason      : {len(empty)}")
    for rel, lineno, reason in empty:
        print(f"  {rel}:{lineno}  reason={reason!r}")
    print()
    print("A skip must carry its reason. A count of skipped tests answers no")
    print("question on its own, and nobody audits three hundred of them.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--reasons", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.scan:
        return scan(args.verbose)
    if args.reasons:
        return reasons()
    ap.error("give --scan or --reasons")


if __name__ == "__main__":
    sys.exit(main())
