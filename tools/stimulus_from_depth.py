#!/usr/bin/env python3
"""Choose a trace stimulus by evaluating the graph's OWN depth expressions.

WHY THIS EXISTS, AND WHAT IT COST NOT TO HAVE IT
------------------------------------------------
On 2026-09-12 the `Wan2.1-VACE` encoder stimulus was raised 9 -> 17 to lift its
temporal axis out of the collision zone {0, 1, 2}. The number 17 was chosen from
the INPUT extent, and the input extent is not where the defect lives. Read from
the re-traced graph, `aten.expand::9`'s CHANNEL slot is

    ((((2 + (s1 - 13)) - 4) // 2 - 1) // 2 + 1) * 384

— a temporal downsampling chain multiplied into a channel count. Evaluate it:

    s1 = 17  ->  1   (the trace saw 1 * 384 = 384 and agreed with itself)
    s1 = 81  ->  17  (the runtime computed 17 * 384 = 6528 and refused)

The stimulus moved the INPUT out of the zone and left that DEPTH at exactly 1,
where `[B, T*C, H, W]` and `[B*T, C, H, W]` are still the same shape. A full
trace -> build -> local -> run chain was spent to learn a number the graph could
have been asked for. The smallest stimulus that clears that depth is 25.

WHAT IT ANSWERS
---------------
Given a graph traced at ANY stimulus, the recorded expressions are functions of
the symbols. So evaluate them:

  * a dim that DEPENDS on the symbol and lands in {0,1,2} is REACHABLE — a larger
    stimulus moves it out, and this tool returns the smallest one that does;
  * a dim that lands in {0,1,2} and reads NO symbol at all is a PLATEAU —
    structurally constant, no stimulus of any axis reaches it, and it needs the
    other gesture (asserting the rule, or distinguishing the two foldings at the
    source);
  * a dim that reads a DIFFERENT symbol is that symbol's business, not this
    one's. Counting it as a plateau inflates the figure with the batch axis —
    2256 of them on the first run of this tool, which is the same over-count the
    input census was corrected for hours earlier.

The plateau count is reported beside the recommendation, never after it: a
stimulus that clears 90% of the depths and leaves a plateau has not corrected the
component, and saying only the first half is how "diagnosed" gets written in the
"corrected" column.

TWO PASSES, AND THAT IS THE POINT
---------------------------------
You cannot evaluate a graph you have not traced. So: trace once at any value,
ask this, re-trace at what it says. The first trace is the instrument, not the
deliverable.

Usage:
    python tools/stimulus_from_depth.py --graph PATH [--symbol time]
    python tools/stimulus_from_depth.py                    # the whole catalogue
    python tools/stimulus_from_depth.py --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

#: The values at which distinct arithmetic rules stop being distinguishable.
#: Identical to the census tools; one definition, three readers.
ZONE = (0, 1, 2)

#: The complete grammar of a recorded dim, measured over the local catalogue on
#: 2026-09-12: `symbol` leaves and binary nodes of exactly these four types.
#: Anything else must RAISE — an evaluator that skips what it does not recognise
#: reports a clean stimulus for a graph it did not read.
BINARY = {
    "mul": lambda a, b: a * b,
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "floordiv": lambda a, b: a // b,
}

#: Axes whose small values are a deliberate, documented project decision rather
#: than a blind spot anyone can act on. Excluded by NAME from the declared symbol
#: table — never by position. (A batch symbol is never a literal 1; 2 is CFG.)
DELIBERATE = {"batch"}


class UnknownNode(ValueError):
    """A dim node whose type the evaluator does not implement."""


def evaluate(node, env: dict):
    """The recorded dim, as a number, under `env` = {symbol id: value}."""
    if isinstance(node, bool):
        raise UnknownNode(f"boolean in a shape: {node!r}")
    if isinstance(node, int):
        return node
    if not isinstance(node, dict):
        raise UnknownNode(f"{type(node).__name__}: {node!r}")
    kind = node.get("type")
    if kind == "symbol":
        sid = node.get("id")
        if sid not in env:
            raise UnknownNode(f"symbol {sid!r} has no value in the environment")
        return env[sid]
    fn = BINARY.get(kind)
    if fn is None:
        raise UnknownNode(f"node type {kind!r}")
    return fn(evaluate(node["left"], env), evaluate(node["right"], env))


def symbols_of(node, out=None) -> set:
    """Which symbol ids this dim actually reads. A dim that reads none is a
    structural constant and no stimulus will move it."""
    out = set() if out is None else out
    if isinstance(node, dict):
        if node.get("type") == "symbol":
            out.add(node.get("id"))
        else:
            for side in ("left", "right"):
                if side in node:
                    symbols_of(node[side], out)
    return out


def declared(graph: dict) -> dict:
    """{symbol id: (name, trace_value)} from the graph's OWN table.

    The declared table, never a positional guess: the positional map names s1
    `latent_h` on every container, and a census built on it was retracted on
    2026-09-12 for reporting a family that did not exist.
    """
    syms = (graph.get("symbolic_context") or {}).get("symbols") or {}
    return {sid: (m.get("name"), m.get("trace_value"))
            for sid, m in syms.items() if isinstance(m, dict)}


def variable_dims(graph: dict):
    """Every (tensor id, dim index, node) that is an EXPRESSION, not a literal.

    Parameters are skipped: a weight extent is a constant by construction, and a
    literal dim that is wrong is a FROZEN dim — a different defect with a
    different gate.
    """
    for tid, meta in (graph.get("tensors") or {}).items():
        if meta.get("is_parameter"):
            continue
        dims = (meta.get("symbolic_shape") or {}).get("dims")
        if not isinstance(dims, list):
            continue
        for i, node in enumerate(dims):
            if isinstance(node, dict):
                yield tid, i, node


def analyse(graph: dict, symbol_id: str, ceiling: int = 512) -> dict:
    """The smallest stimulus for `symbol_id` that leaves no REACHABLE depth in
    the collision zone, plus the plateau that no stimulus reaches."""
    table = declared(graph)
    base = {sid: trace for sid, (_, trace) in table.items()
            if isinstance(trace, int)}
    if symbol_id not in base:
        raise KeyError(f"{symbol_id} is not a declared symbol with a trace value")

    reachable, plateau, elsewhere = [], [], 0
    for tid, i, node in variable_dims(graph):
        reads = symbols_of(node)
        try:
            at_trace = evaluate(node, base)
        except UnknownNode as exc:
            raise UnknownNode(f"{tid} dim{i}: {exc}") from exc
        if symbol_id in reads:
            reachable.append((tid, i, node))
        elif reads:
            # Another axis owns it. Its own run of this tool adjudicates it.
            elsewhere += at_trace in ZONE
        elif at_trace in ZONE:
            plateau.append((tid, i, at_trace))

    def clear_at(value: int) -> bool:
        env = dict(base, **{symbol_id: value})
        for _, _, node in reachable:
            v = evaluate(node, env)
            if v in ZONE or v < 0:     # a negative extent is not a shape
                return False
        return True

    chosen = next((v for v in range(1, ceiling + 1) if clear_at(v)), None)
    at_trace_ok = clear_at(base[symbol_id])
    return {
        "symbol": symbol_id,
        "name": table[symbol_id][0],
        "trace_value": base[symbol_id],
        "trace_value_is_clear": at_trace_ok,
        "recommended": chosen,
        "reachable_dims": len(reachable),
        "plateau_dims": len(plateau),
        "in_zone_under_another_symbol": elsewhere,
        "plateau_sample": [f"{t} dim{i}={v}" for t, i, v in plateau[:5]],
    }


def _rows(graph: dict, only: str | None):
    for sid, (name, trace) in sorted(declared(graph).items()):
        if not isinstance(trace, int):
            continue
        if only and name != only:
            continue
        if not only and (name or "").lower() in DELIBERATE:
            continue
        yield analyse(graph, sid)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--graph", help="one component graph.json")
    ap.add_argument("--root", default=os.environ.get(
        "NEUROBRIX_CACHE", str(Path.home() / ".neurobrix" / "cache")))
    ap.add_argument("--symbol", help="restrict to this symbol NAME (e.g. time)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.graph:
        paths = [Path(args.graph)]
    else:
        paths = sorted(Path(args.root).glob("*/components/*/graph.json"))
    if not paths:
        print("REFUSED: no component graph to read. An empty analysis is not a "
              "clean bill of health.", file=sys.stderr)
        return 1

    out, moved, plateaued = [], 0, 0
    for path in paths:
        key = f"{path.parents[2].name}/{path.parent.name}" if not args.graph else str(path)
        try:
            graph = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            print(f"  {key}: UNREADABLE — {exc}", file=sys.stderr)
            continue
        for row in _rows(graph, args.symbol):
            row["component"] = key
            out.append(row)
            if not row["trace_value_is_clear"]:
                moved += 1
            if row["plateau_dims"]:
                plateaued += 1

    if args.json:
        print(json.dumps(out, indent=1))
        return 0

    need = [r for r in out if not r["trace_value_is_clear"] or r["plateau_dims"]]
    print(f"Stimulus from depth — {len(paths)} graph(s), {len(out)} symbol(s)\n")
    for r in sorted(need, key=lambda r: -r["plateau_dims"])[:40]:
        rec = r["recommended"]
        verdict = ("clear at its trace value" if r["trace_value_is_clear"]
                   else (f"RAISE {r['trace_value']} -> {rec}" if rec
                         else f"NO stimulus <= 512 clears it"))
        print(f"  {r['component']}  '{r['name']}' (s={r['trace_value']})")
        print(f"      {verdict}   [{r['reachable_dims']} reachable dims]")
        if r["plateau_dims"]:
            print(f"      PLATEAU: {r['plateau_dims']} dim(s) in {ZONE} that read no "
                  f"symbol at all — no stimulus of any axis reaches them")
            for s in r["plateau_sample"]:
                print(f"         {s}")
    print(f"\n  symbols whose own trace value sits in the zone at some depth: {moved}")
    print(f"  symbols with a structural plateau beside them: {plateaued}")
    print("\n  A recommendation without its plateau is half an answer: the stimulus "
          "corrects\n  what it reaches, and the rest needs the rule asserted at the "
          "source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
