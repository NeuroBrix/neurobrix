#!/usr/bin/env python3
"""Is a dimension frozen? Read the ARGUMENTS, never `output_shapes`.

    tools/frozen_dim_report.py <model> [component]

`output_shapes` records what the shapes WERE at trace time; literals there are
expected and carry no claim. The expression the runtime evaluates lives in
`attributes.args`. D-MOCHI was filed from the first field and cost mochi a 2.7 h
re-trace that could not have changed anything — the op it named carries `s1*3`
in its arguments, in both containers.

Full rule: docs/reference/a-frozen-dimension-is-read-from-the-arguments.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CACHE = Path.home() / ".neurobrix" / "cache"

#: Ops whose arguments carry SIZES. Anything else takes no sizes and proves
#: nothing: `aten::permute` takes a permutation order, and flagging it produced
#: 87 false positives the first time this was measured.
SIZE_OPS = {"aten::view", "aten::_unsafe_view", "aten::reshape", "aten::expand",
            "aten::broadcast_to", "aten::repeat"}


def _is_dynamic(node) -> bool:
    """A size argument that needs no symbol because it is already dynamic.

    `-1` asks the framework to INFER the dimension from the element count, which
    is the most dynamic form a view can take. A scan that calls it "literal"
    because it holds no `symbol` node reports a freeze where there is none —
    this tool did exactly that on mochi's `aten.view::36`, whose argument is
    `[-1, 768]`, minutes after being written to stop that class of mistake.
    """
    if isinstance(node, dict):
        vals = node.get("value") if node.get("type") == "list" else None
        if isinstance(vals, list):
            return any(v == -1 for v in vals if isinstance(v, int))
        return any(_is_dynamic(v) for v in node.values())
    if isinstance(node, list):
        return any(_is_dynamic(v) for v in node)
    return False


def _mentions_symbol(node) -> bool:
    if isinstance(node, dict):
        if node.get("type") == "symbol":
            return True
        return any(_mentions_symbol(v) for v in node.values())
    if isinstance(node, list):
        return any(_mentions_symbol(v) for v in node)
    return False


def report(model: str, only: str | None = None) -> int:
    base = CACHE / model / "components"
    if not base.is_dir():
        print(f"no components under {base}")
        return 2
    frozen_total = 0
    for comp_dir in sorted(base.iterdir()):
        if only and comp_dir.name != only:
            continue
        graph = comp_dir / "graph.json"
        if not graph.is_file():
            continue
        d = json.loads(graph.read_text())
        ops = d.get("ops") or {}
        sym = lit = 0
        frozen = []
        for uid, op in ops.items():
            if op.get("op_type") not in SIZE_OPS:
                continue
            args = (op.get("attributes") or {}).get("args")
            if _mentions_symbol(args) or _is_dynamic(args):
                sym += 1
            else:
                lit += 1
                frozen.append((uid, op.get("op_type"), op.get("output_shapes")))
        frozen_total += lit
        print(f"  {model}/{comp_dir.name}: size-ops {sym + lit} — "
              f"symbolic-or-dynamic {sym}, LITERAL {lit}")
        for uid, t, shapes in frozen[:5]:
            print(f"      frozen: {uid} [{t}] output_shapes={json.dumps(shapes)[:70]}")
    print(f"  read from attributes.args, never from output_shapes — "
          f"{frozen_total} literal size argument(s) in total")
    return 1 if frozen_total else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(report(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
