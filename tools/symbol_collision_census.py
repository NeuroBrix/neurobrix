#!/usr/bin/env python3
"""Where would a symbolic-shape defect be INVISIBLE? — a census of trace values.

This does not look for defects. It looks for the places a defect could not be
seen, which is a different and larger set.

WHY IT EXISTS
-------------
A symbolic shape rule is validated by exactly one comparison: does the expression
it produces reproduce the extent witnessed at trace? That comparison is made at a
SINGLE point of the symbol's domain. Any error that happens to vanish at that
point is invisible to it permanently, because a check at one point cannot separate
two functions that agree at that point (register entry 29).

The case that engraved it, 2026-09-12: `CogVideoX-5b-I2V/vae_encoder`'s causal
temporal pad recorded `3*s` where the truth is `s + 2`. Its trace value was 1, and
`3x1 == 1+2 == 3`. Seven resnet blocks then compounded it to `2187*s - 2184`,
still exactly 3 at s=1, and Prism asked for 944 GB of activations for a component
whose weights are 822 MB. Four independent structural invariants all passed — all
evaluated at that one point.

**Of 27 five-dimensional components in the local catalogue, exactly ONE was traced
with its temporal axis at 1, and it is the only one carrying the defect.** That is
why this is a census and not an anecdote: the population of blind spots is
enumerable, and it is small.

THE COLLISION CLASSES
---------------------
An axis traced at value `v` cannot distinguish these rules:

  v == 0   everything collapses. No rule is observable.
  v == 1   multiplication, addition, exponentiation and identity all agree:
           k*s == s+(k-1) == s**n == s. THE WORST VALUE, and the one a frugal
           stimulus reaches most easily (an I2V encoder conditions on ONE image,
           so its temporal axis is genuinely 1 at trace).
  v == 2   doubling and squaring agree (2*2 == 2**2 == 4); so do s+2 and 2*s.
  v == w   where w is a WEIGHT dimension of the same component: a dim bound to
           the wrong quantity reproduces the trace anyway. This is the
           trace-value collision the project already guards at build time; the
           census reports it because a guard that fired is not the same as an
           axis that is safe.

The answer this gives is not "here is a bug". It is "here is where a bug would
be invisible" — which is what tells you whether a case is a case or a population.

Usage:
    python tools/symbol_collision_census.py                 # the local catalogue
    python tools/symbol_collision_census.py --root DIR      # somewhere else
    python tools/symbol_collision_census.py --json          # machine-readable
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

#: Values at which distinct arithmetic rules become indistinguishable, with the
#: rules each one hides. Data, not prose: the report names what a reader would
#: otherwise have to re-derive.
ARITHMETIC_COLLISIONS = {
    0: "every rule collapses (k*s, s+k, s**n all vanish or fix)",
    1: "k*s == s+(k-1) == s**n == s — multiplication, addition, power, identity",
    2: "2*s == s**2 == s+2 — doubling, squaring and a +2 offset",
}


def declared_symbols(graph: dict) -> dict:
    """{symbol id: (name, trace_value)} as the graph's own table declares them."""
    syms = (graph.get("symbolic_context") or {}).get("symbols") or {}
    out = {}
    if isinstance(syms, dict):
        for sid, meta in syms.items():
            if isinstance(meta, dict):
                out[sid] = (meta.get("name"), meta.get("trace_value"), meta.get("source"))
    return out


def weight_extents(graph: dict) -> set:
    """Every extent appearing in a PARAMETER's shape.

    A symbol whose trace value equals one of these can be bound to the wrong
    quantity and still reproduce the trace — the collision the tracer's own
    guard exists for.
    """
    out = set()
    for meta in (graph.get("tensors") or {}).values():
        if not meta.get("is_parameter"):
            continue
        for extent in (meta.get("shape") or []):
            if isinstance(extent, int) and extent > 2:
                out.add(extent)
    return out


def census_one(path: Path, show_all: bool = False) -> tuple:
    try:
        graph = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}], []
    weights = weight_extents(graph)
    rows, accepted = [], []
    for sid, (name, trace, source) in sorted(declared_symbols(graph).items()):
        flags = []
        if isinstance(trace, int) and trace in ARITHMETIC_COLLISIONS:
            flags.append(("arithmetic", ARITHMETIC_COLLISIONS[trace]))
        if isinstance(trace, int) and trace in weights:
            flags.append(("weight-extent",
                          f"{trace} is also a parameter extent in this component"))
        # THE BATCH AXIS IS A DELIBERATE, DOCUMENTED CHOICE at every small
        # value it takes, and not a blind spot anyone can act on. The project
        # REQUIRES batch to stay symbolic at a single item ("a batch symbol is
        # never a literal 1" — a view that froze it cost 14 containers on
        # 2026-08-29, invisible to byte gates precisely BECAUSE batch was 1),
        # and 2 is the CFG batch, equally deliberate. Listing it beside the
        # actionable axes buries them: of 293 flagged axes it is 58, and of the
        # 57 axes at trace 2 nearly all are this one class.
        #
        # It is COUNTED and named, never silently dropped, and `--all` shows it
        # for anyone re-examining the decision itself rather than its consequences.
        if (name or "").lower() == "batch" and not show_all:
            accepted.append(sid)
            continue
        if flags:
            rows.append({"symbol": sid, "name": name, "trace": trace,
                         "source": source, "flags": flags})
    return rows, accepted


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.environ.get(
        "NEUROBRIX_CACHE", str(Path.home() / ".neurobrix" / "cache")))
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--all", action="store_true",
                    help="also list the batch-at-1 axes, which are a deliberate choice")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"REFUSED: no catalogue at {root}", file=sys.stderr)
        return 1

    graphs = sorted(root.glob("*/components/*/graph.json"))
    if not graphs:
        # A census that examined nothing must not read as one that found nothing.
        print(f"REFUSED: {root} holds no component graph. A census over zero "
              f"graphs is not a clean census.", file=sys.stderr)
        return 1

    findings, clean, per_class = {}, 0, collections.Counter()
    accepted_total = 0
    for path in graphs:
        key = f"{path.parents[2].name}/{path.parent.name}"
        rows, accepted = census_one(path, show_all=args.all)
        accepted_total += len(accepted)
        if rows:
            findings[key] = rows
            for row in rows:
                for cls, _ in row.get("flags", []):
                    per_class[cls] += 1
        else:
            clean += 1

    if args.json:
        print(json.dumps({"root": str(root), "graphs": len(graphs),
                          "clean": clean, "findings": findings}, indent=1))
        return 0

    print(f"Symbol-collision census — {len(graphs)} component graph(s) under {root}")
    print(f"{clean} carry no ACTIONABLE symbol at a collision value.")
    if accepted_total:
        print(f"{accepted_total} batch axis/axes traced at 1 are excluded: that is a "
              f"deliberate\nproject decision (a batch symbol is never a literal 1), "
              f"not a blind spot to act on.\nRe-examine it with --all.")
    print()
    if not findings:
        print("No axis is traced at a value where two rules become "
              "indistinguishable. Nothing here is a clean bill of health for the "
              "rules themselves — it says a defect in them would be VISIBLE.")
        return 0

    for key in sorted(findings):
        print(f"  {key}")
        for row in findings[key]:
            src = f"  <- {row['source']}" if row.get("source") else ""
            print(f"      {row['symbol']} '{row['name']}' traced at {row['trace']}{src}")
            for cls, why in row["flags"]:
                print(f"          [{cls}] {why}")
    print(f"\n  components with at least one blind axis: {len(findings)} of {len(graphs)}")
    print(f"  containers affected: "
          f"{len({k.split('/')[0] for k in findings})}")
    for cls, n in per_class.most_common():
        print(f"  {cls}: {n} axis/axes")
    print("\n  This is where a defect would be INVISIBLE, not where one is. An axis "
          "listed here\n  needs its rule asserted structurally, or a re-trace at a "
          "value outside the\n  collision — a test at the flagged value is green for "
          "the reason that blinds it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
