#!/usr/bin/env python3
"""Which local containers draw random numbers inside their graph, and which
request reaches the component that draws.

    tools/rng_census.py                 # the census, one line per container

The class is bounded (D-RNG-DRAW-UNARMED-IN-A-FLOW, filed 2026-09-11): an RNG
op in a component's graph is a draw the engine must make from the stream it
arms at the executor, or two runs of the same request differ. The census was
taken by hand for the debt; this is the brick, so the guard that runs each
such container twice reads the same list the debt did, and a container added
tomorrow with a draw is in the guard the day it lands.

`request_reaching(topology, component)` says, from the container's own
topology, which request exercises the component: `"default"` when it is in
the flow's order or its audio stages, `"speech"` when it belongs to a speech
leg (`flow.speech.components`), which the CLI reaches with `--mode audio`. A
component none of those name is `None` — the guard refuses to claim it ran.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional

RNG_OPS = {"aten::rand", "aten::randn", "aten::rand_like", "aten::randn_like", "aten::normal",
           "aten::bernoulli", "aten::poisson", "aten::exponential", "aten::uniform"}
CACHE = Path.home() / ".neurobrix" / "cache"


def rng_ops_of(graph_path: Path) -> Dict[str, int]:
    """{op_type: count} of the RNG ops in one component graph."""
    try:
        doc = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: Dict[str, int] = {}
    ops = doc.get("ops") or []
    ops = ops.values() if isinstance(ops, dict) else ops     # some graphs key their ops by uid
    for op in ops:
        t = op.get("op_type") if isinstance(op, dict) else None
        if t in RNG_OPS:
            out[t] = out.get(t, 0) + 1
    return out


def containers_with_rng_ops(cache: Path = CACHE) -> Dict[str, Dict[str, Dict[str, int]]]:
    """{container: {component: {op_type: count}}} over the local cache."""
    out: Dict[str, Dict[str, Dict[str, int]]] = {}
    if not cache.exists():
        return out
    for sub in sorted(cache.iterdir()):
        if not sub.is_dir() or sub.name.endswith("-backup") or not (sub / "manifest.json").exists():
            continue
        for g in sorted(sub.rglob("graph.json")):
            ops = rng_ops_of(g)
            if ops:
                out.setdefault(sub.name, {})[g.parent.name] = ops
    return out


def request_reaching(topology: dict, component: str) -> Optional[str]:
    """'default' | 'speech' | None — read from the topology, never from a name."""
    flow = topology.get("flow") or {}
    if component in (flow.get("order") or []):
        return "default"
    audio = flow.get("audio") or {}
    if component in [s.get("component") for s in (audio.get("stages") or [])]:
        return "default"
    speech = flow.get("speech") or {}
    if component in (speech.get("components") or {}).values():
        return "speech"
    return None


def main() -> int:
    census = containers_with_rng_ops()
    for name, comps in census.items():
        topo = json.loads((CACHE / name / "topology.json").read_text(encoding="utf-8"))
        parts = [f"{c}: " + ", ".join(f"{o.split('::')[1]} x{n}" for o, n in ops.items())
                 + f" [{request_reaching(topo, c) or 'UNREACHED by any request the topology names'}]"
                 for c, ops in comps.items()]
        print(f"{name}: " + "; ".join(parts))
    print(f"{len(census)} container(s) draw inside their graph")
    return 0


if __name__ == "__main__":
    sys.exit(main())
