#!/usr/bin/env python3
"""A harness metric that reads a key its brick never emits returns a verdict about nothing.

WHY THIS EXISTS
---------------
`tools/vendor_correctness_cell.py::m_psnr_db` read `d.get("psnr", 0.0)` while
`tools/image_fidelity.py` emits `psnr_db`. The lookup fell back to 0.0 on every
render, so **no image cell could ever report AGREES** whatever the two pictures
were — and it reported DIVERGES on two images a human eyeballed as correct.

The repository already had vacuity gates, and they did not see it: they assert
that a numeric BOUND is not too loose, and they live in `tests/`. This defect was
not a loose bound and not in a test. It was a harness reading a key that does not
exist, which is a gate that cannot pass — the strongest form of vacuity there is,
because it has no true branch at all.

This audit covers the other half: every campaign harness, checked against what
its bricks actually emit.

HOW IT DECIDES
--------------
1. For each brick, the keys it really emits — collected from the dict literals it
   serialises (`json.dumps({...})`, `json.dump({...}, ...)`, `write_text(json.dumps(...))`).
   Only literal keys are collected; a brick that builds its dict dynamically is
   reported as UNKNOWN rather than guessed at, because a wrong emit-set would
   manufacture false findings.
2. For each harness function that invokes a brick by name AND parses JSON, the
   keys it reads out of that JSON — `d["k"]`, `d.get("k")`, `d.get("k", default)`.
3. A key read but never emitted is a finding. A `.get()` with a DEFAULT is the
   severe form: it silently returns the default and the verdict is computed on
   it, which is what happened. A bare `d["k"]` at least raises.

The audit is deliberately conservative: it reports only where it can name both
sides. What it cannot resolve it says it cannot resolve, and never turns that
into a pass.
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import sys
from pathlib import Path


def _dict_keys(node) -> set:
    """Literal keys of a dict literal, nested ones included."""
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Dict):
            for k in sub.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    out.add(k.value)
    return out


def emitted_keys(path: Path) -> tuple:
    """(keys, complete) — the literal keys this brick serialises.

    A brick rarely writes `json.dumps({...})` inline. The common shape is a
    function that RETURNS the payload as a dict literal and a `main()` that
    dumps it — `tools/image_fidelity.py::compare` is exactly that, and reading
    only the inline form made this audit blind to the very brick whose consumer
    carried the defect it was written for.
    """
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:
        return set(), False
    keys, complete = set(), False
    serialises = any(
        isinstance(n, ast.Call)
        and ((isinstance(n.func, ast.Attribute) and n.func.attr in ("dumps", "dump"))
             or (isinstance(n.func, ast.Name) and n.func.id in ("dumps", "dump")))
        for n in ast.walk(tree))
    if serialises:
        for n in ast.walk(tree):
            if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict):
                keys |= _dict_keys(n.value)
                complete = True
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = (fn.attr if isinstance(fn, ast.Attribute) else
                fn.id if isinstance(fn, ast.Name) else "")
        if name not in ("dumps", "dump"):
            continue
        if not node.args:
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Dict):
            complete = True
            # Nested literals count too: a brick emitting
            # {"kernels": [{"module": ...}]} does emit "module", one level in,
            # and a consumer reading it is reading something real.
            keys |= _dict_keys(arg)
        else:
            # a name or a comprehension: the brick builds its payload elsewhere
            pass
    return keys, complete


def read_keys(fn_node: ast.AST) -> list:
    """Every literal key this function reads out of a dict, with how it reads it."""
    out = []
    for n in ast.walk(fn_node):
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) \
                and isinstance(n.slice.value, str) \
                and isinstance(getattr(n, "ctx", None), ast.Load):
            # Load only. `d["pass"] = ...` WRITES a key into the payload — a
            # harness enriching what it read — and counting that as a read of an
            # absent key is this auditor crying wolf about its own subject.
            out.append({"key": n.slice.value, "how": "[]", "line": n.lineno})
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and n.func.attr == "get" and n.args:
            k = n.args[0]
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                out.append({"key": k.value,
                            "how": "get+default" if len(n.args) > 1 else "get",
                            "line": n.lineno})
    return out


def function_nodes(tree: ast.AST):
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield n


def audit(harnesses: list, bricks: dict) -> list:
    findings = []
    for h in harnesses:
        try:
            tree = ast.parse(Path(h).read_text(errors="replace"))
        except SyntaxError:
            continue
        for fn in function_nodes(tree):
            src = ast.dump(fn)
            # which bricks does this function invoke, by file name?
            used = [b for b in bricks
                    if Path(b).resolve() != Path(h).resolve()
                    and (Path(b).name in src or Path(b).stem in src)]
            if not used:
                continue
            if "json" not in src or ("loads" not in src and "load" not in src):
                continue
            for b in used:
                keys, complete = bricks[b]
                if not complete:
                    findings.append({"harness": h, "function": fn.name, "brick": b,
                                     "verdict": "UNKNOWN-EMIT-SET", "key": None,
                                     "line": fn.lineno,
                                     "detail": "the brick builds its payload dynamically; "
                                               "this audit will not guess its keys"})
                    continue
                for r in read_keys(fn):
                    if r["key"] in keys:
                        continue
                    # keys that plainly belong to another dict in the function
                    if r["key"] in ("error", "output", "verdict", "reason", "id",
                                    "family", "model", "vendor", "compare", "metric",
                                    "bound", "gate", "request", "kind", "ref", "venv",
                                    "prompt", "seed", "steps", "guidance", "height",
                                    "width", "audio", "frames", "max_tokens",
                                    "temperature", "blind_to", "driven_by", "cells",
                                    "precision", "input_image", "latents"):
                        continue
                    findings.append({
                        "harness": h, "function": fn.name, "brick": b,
                        "key": r["key"], "how": r["how"], "line": r["line"],
                        "verdict": "SILENT-DEFAULT" if r["how"] == "get+default" else "ABSENT-KEY",
                        "detail": f"{Path(b).name} emits {sorted(keys)}",
                    })
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harnesses", nargs="*", default=None,
                    help="default: tools/*.py + benchmarks/harness/*.py")
    ap.add_argument("--bricks", nargs="*", default=None,
                    help="default: every tools/*.py that serialises a dict literal")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    harnesses = a.harnesses or sorted(
        glob.glob("tools/*.py") + glob.glob("benchmarks/harness/*.py"))
    brick_files = a.bricks or sorted(glob.glob("tools/*.py"))
    bricks = {}
    for b in brick_files:
        keys, complete = emitted_keys(Path(b))
        if keys or complete:
            bricks[b] = (keys, complete)

    findings = audit(harnesses, bricks)
    severe = [f for f in findings if f["verdict"] == "SILENT-DEFAULT"]
    absent = [f for f in findings if f["verdict"] == "ABSENT-KEY"]
    unknown = [f for f in findings if f["verdict"] == "UNKNOWN-EMIT-SET"]

    for f in severe + absent:
        print(f"  {f['verdict']:15s} {f['harness']}:{f['line']}  {f['function']}()")
        print(f"      reads {f['key']!r} via {f['how']} — {f['detail'][:120]}")
    print(f"\n{len(harnesses)} harness file(s) against {len(bricks)} brick(s): "
          f"{len(severe)} silent-default, {len(absent)} absent-key, "
          f"{len(unknown)} brick(s) whose emit set could not be read")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(findings, indent=1))
        print(f"written: {a.out}")
    # a silent default is the form that returns a verdict about nothing
    return 1 if severe else 0


if __name__ == "__main__":
    sys.exit(main())
