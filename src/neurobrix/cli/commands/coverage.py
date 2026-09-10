"""`neurobrix coverage` — which installed containers actually reach a symbol.

WHY THIS COMMAND EXISTS

A test that exercises nothing is green. That is the most expensive class of
defect this project has met, and it keeps arriving in the same disguise: a
verdict everybody reads as "correct" which is really "never ran".

Three of them, in three days:

  * A benchmark harness asked every row for `--temperature 0`, fourteen times.
    The sampling paths above greedy were never once entered by the campaign
    that was supposed to cover them.
  * The flash-attention path was pruned by a clamp that returned `(None, None)`
    for every unshipped profile, so on those cards nothing was pruned and
    nothing was flash — measured only when a user's A40 refused to start.
  * Nineteen kernel edits landed with a plan to validate them on the zoo. A
    census of the 56 local containers found `aten::argmin`, `aten::min` and
    `aten::var` in **none** of them: three of the nineteen sites cannot be
    reached by any model run, so that plan's zoo cell would have been green
    having touched nothing.

Each of those was one command away from being obvious. So the census stops
being a thing somebody computes when they remember, and becomes a thing anyone
can ask.

    neurobrix coverage aten::tril        # who reaches it, and how many
    neurobrix coverage --rarest 25       # what the catalogue barely exercises
    neurobrix coverage --unreached       # what it never exercises at all
    neurobrix coverage --field temperature

WHAT IT CANNOT ANSWER, SAID HERE RATHER THAN DISCOVERED LATER

This reads the containers, so it answers questions about the GRAPH: which ATen
op a model carries, which field its metadata declares. It cannot answer a
question about a RUNTIME DECISION — whether the flash branch or the math branch
was taken, which autotune config was seated, whether a fallback fired. Those are
chosen while running, they are not nodes in a graph, and asking this command
about them would produce a confident wrong answer. That class needs its own
instrument, on the execution path.

A count of zero here means "no installed container carries it", not "dead code":
a kernel can be reached by a model nobody has built yet, and the hub holds more
than this machine does.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

#: Extracted with a scan rather than `json.load`: a single MoE graph carries
#: 44,634 ops, and the whole catalogue parses in seconds this way against
#: minutes that way. The key is part of the NBX contract (R18), so it does not
#: drift under us — and `test_coverage_reads_the_container_contract` pins it.
_OP_TYPE = re.compile(rb'"op_type"\s*:\s*"([^"]+)"')


def _cache_root() -> Path:
    import os
    return Path(os.environ.get("NEUROBRIX_CACHE",
                               Path.home() / ".neurobrix" / "cache"))


def _containers(root: Path):
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists())


def _ops_of(container: Path) -> set:
    found = set()
    for graph in container.rglob("graph.json"):
        try:
            data = graph.read_bytes()
        except OSError:
            continue
        found.update(m.group(1).decode() for m in _OP_TYPE.finditer(data))
    return found


def _index(root: Path):
    """{op_type: {container names}} over every installed container."""
    index = defaultdict(set)
    containers = _containers(root)
    for c in containers:
        for op in _ops_of(c):
            index[op].add(c.name)
    return index, [c.name for c in containers]


def _fields_of(container: Path, key: str):
    """Every value declared under `key`, anywhere in the container metadata."""
    hits = []
    for meta in list(container.glob("*.json")) + list(container.rglob("runtime/*.json")):
        try:
            doc = json.loads(meta.read_text())
        except (OSError, ValueError):
            continue
        stack = [doc]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for k, v in node.items():
                    if k == key:
                        hits.append(v)
                    elif isinstance(v, (dict, list)):
                        stack.append(v)
            elif isinstance(node, list):
                stack.extend(x for x in node if isinstance(x, (dict, list)))
    return hits


def cmd_coverage(args) -> int:
    root = _cache_root()
    containers = _containers(root)
    if not containers:
        print(f"No installed container under {root}.", file=sys.stderr)
        print("This command measures what THIS machine holds; an empty cache "
              "makes every answer zero for the wrong reason.", file=sys.stderr)
        return 1

    if args.field:
        total = 0
        for c in containers:
            hits = _fields_of(c, args.field)
            if hits:
                total += 1
                shown = sorted({json.dumps(h, sort_keys=True) for h in hits})
                print(f"  {c.name}: {', '.join(shown[:4])}"
                      + (f"  (+{len(shown) - 4} more)" if len(shown) > 4 else ""))
        print(f"\n`{args.field}` declared by {total} of {len(containers)} containers.")
        return 0

    index, names = _index(root)

    if args.unreached or args.rarest:
        counts = Counter({op: len(who) for op, who in index.items()})
        if args.unreached:
            # What the catalogue never exercises can only be listed against a
            # list of what exists — and the classification table is where the
            # engine says which ops it claims to handle.
            claimed = _claimed_ops()
            missing = sorted(op for op in claimed if not index.get(f"aten::{op}"))
            print(f"Ops the engine classifies but NO installed container carries "
                  f"({len(missing)} of {len(claimed)}):\n")
            for op in missing:
                print(f"  aten::{op}")
            print("\nA kernel for one of these cannot be validated by running a "
                  "model. It needs a direct kernel test, or the verdict is a "
                  "gate that measured nothing.")
            return 0
        print(f"The {args.rarest} ops carried by the fewest containers "
              f"(of {len(names)}):\n")
        for op, n in sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))[:args.rarest]:
            who = sorted(index[op])
            tail = ", ".join(who[:3]) + (f", +{len(who) - 3}" if len(who) > 3 else "")
            print(f"  {n:3d}  {op:<44} {tail}")
        return 0

    if not args.symbol:
        print(f"{len(index)} distinct ATen ops across {len(names)} containers "
              f"in {root}.")
        print("Give a symbol (`neurobrix coverage aten::tril`), or ask for "
              "`--rarest N` / `--unreached` / `--field KEY`.")
        return 0

    symbol = args.symbol if "::" in args.symbol else f"aten::{args.symbol}"
    who = sorted(index.get(symbol, ()))
    print(f"{symbol}: {len(who)} of {len(names)} installed containers\n")
    for name in who:
        print(f"  {name}")
    if not who:
        print("  (none)\n")
        print("No installed container carries this op, so no model run can "
              "exercise the kernel that serves it. That is a statement about "
              "this machine's cache, not proof the kernel is dead — but any "
              "plan that validates it by running the zoo is a plan that "
              "validates nothing.")
    return 0


def _claimed_ops():
    """The op names the engine's classification table declares it handles."""
    from neurobrix.kernels import classification
    claimed = set()
    for attr in dir(classification):
        value = getattr(classification, attr)
        if isinstance(value, (set, frozenset, list, tuple)) and attr.isupper():
            claimed.update(v for v in value if isinstance(v, str) and "::" not in v)
    return claimed
