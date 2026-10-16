"""A component's temporal stream, as Prism reads and prices it (decision of 2026-10-16 18:40 / 21:30).

Forge derives, from a causal video component's traced clip graph, two graphs and a declaration
(`components/<c>/stream_first.json`, `stream_next.json`, `stream.json`; Forge `tools/derive_temporal_stream.py`): the
first chunk's graph (the clip graph plus the states it leaves) and the graph of every later chunk (the states in and
out). Run chunk by chunk they ARE the clip graph — proven in Forge, float64 — where no op is global over time; where one
is (a group norm over [C, T, H, W]), they are the vendor's own chunked function, and the declaration carries the
vendor's schedule with its source (`vendor_chunks`).

Prism reads here: which chunk lengths are exact (the conditions the declaration writes: a strided window's alignment, a
leading slice's offset — never a model name), the schedule a request runs (the vendor's when declared, else the largest
exact chunk whose activations fit), and a chunk's price (the activation profiler on the chunk graph, its time symbol
bound to the chunk). Pure Python, no torch (R33): the plan reads files and arithmetic.

ZERO FALLBACK: a declaration whose clip-graph digest is not the container's graph is refused by name (a stale stream
would run another function); a condition kind not read here is refused.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from neurobrix.core.runtime import symexpr as _symexpr


class StreamDeclarationError(RuntimeError):
    """A container's stream declaration is not the one its graph was derived into."""


def load(cache_path: Path, comp: str) -> Optional[Dict[str, Any]]:
    """The component's stream declaration with its two graphs (`decl["dags"]["first" | "next"]`), or None when the
    container carries none. Refused by name when it was derived from another graph than the component's."""
    d = Path(cache_path) / "components" / comp
    path = d / "stream.json"
    if not path.exists():
        return None
    decl = json.loads(path.read_text())
    if decl.get("format") != "temporal_stream" or decl.get("version") != 1:
        raise StreamDeclarationError(f"{path}: format {decl.get('format')!r} version {decl.get('version')!r} is not "
                                     f"the temporal_stream v1 this engine reads")
    sha = hashlib.sha256((d / "graph.json").read_bytes()).hexdigest()
    if sha != decl.get("clip_graph_sha256"):
        raise StreamDeclarationError(
            f"{path}: derived from a graph of sha256 {decl.get('clip_graph_sha256')}, the component's graph.json is "
            f"{sha} — a stale stream runs another function; re-derive it (Forge tools/derive_temporal_stream.py)")
    decl["dags"] = {k: json.loads((d / name).read_text()) for k, name in decl["graphs"].items()}
    return decl


def _value(node: Any, symbol: str, length: int) -> int:
    def sym(sid, _n):
        if sid != symbol:
            raise _symexpr.ExpressionError(f"a stream condition reads symbol {sid}, not the chunk's {symbol}")
        return int(length)
    return int(_symexpr.evaluate(node, sym, _symexpr.SHAPE))


def condition_holds(c: Dict[str, Any], symbol: str, length: int) -> bool:
    v = _value(c["extent"], symbol, length)
    kind = c["kind"]
    if kind == "ge":
        return v >= c["bound"]
    if kind == "window":
        return v >= c["k"] and (v - c["k"]) % c["s"] == 0
    if kind == "multiple":
        return v % c["s"] == 0
    raise StreamDeclarationError(f"a stream condition of kind {kind!r} is not read by this engine")


def broken(decl: Dict[str, Any], lengths: List[int]) -> List[str]:
    """What a schedule (first chunk, then the next ones) breaks, named; [] when every chunk is exact."""
    out = []
    for i, n in enumerate(lengths):
        graph = "first" if i == 0 else "next"
        for c in decl["conditions"]:
            if c["graph"] == graph and not condition_holds(c, decl["symbol"], n):
                out.append(f"chunk {i} of {n} frames breaks a {c['kind']} condition at {c['op']}")
    return out


def vendor_schedule(decl: Dict[str, Any], frames: int) -> List[int]:
    """The vendor's chunk lengths over `frames` input frames (`vendor_chunks`: chunks of `length`, the remainder in
    the first — CogVideoX's `_decode`)."""
    vc = decl["vendor_chunks"]
    n = int(vc["length"])
    if vc.get("remainder", "first") != "first":
        raise StreamDeclarationError(f"vendor_chunks remainder {vc.get('remainder')!r} is not read by this engine")
    if frames <= n:
        return [frames]
    return [n + frames % n] + [n] * (frames // n - 1)


def exact_schedule(decl: Dict[str, Any], frames: int, next_len: int) -> Optional[List[int]]:
    """A schedule of `frames` input frames whose next chunks are `next_len` long (the last one shorter when it is
    exact too), its first chunk the shortest exact one that leaves an exact rest; None when there is none."""
    for first in range(1, frames + 1):
        if broken(decl, [first]):
            continue
        rest = frames - first
        if rest == 0:
            return [first]
        k, r = divmod(rest, next_len)
        sched = [first] + [next_len] * k + ([r] if r else [])
        if not broken(decl, sched):
            return sched
    return None


def choose(decl: Dict[str, Any], frames: int, fits: Callable[[List[int]], bool]) -> Optional[List[int]]:
    """The schedule a request runs: the vendor's when declared (whether it fits is the caller's to judge), else the
    exact schedule of the LONGEST next chunk that `fits` — fewest chunks, the optimum (R5). None: no exact schedule
    fits."""
    if decl.get("vendor_chunks"):
        return vendor_schedule(decl, frames)
    for next_len in range(frames, 0, -1):
        sched = exact_schedule(decl, frames, next_len)
        if sched is not None and fits(sched):
            return sched
    return None


def state_bytes(decl: Dict[str, Any], bind: Dict[str, int], dtype_bytes: int) -> int:
    """The carried states' bytes at a binding of the spatial symbols (the states hold no time symbol)."""
    total = 0
    for s in decl["states"]:
        n = 1
        for d in s["dims"]:
            n *= int(_symexpr.evaluate(d, lambda sid, _n: bind[sid], _symexpr.SHAPE))
        total += n * dtype_bytes
    return total


def output_frames(decl: Dict[str, Any], kind: str, length: int) -> int:
    """The output's time extent the `kind` chunk graph gives at a chunk of `length` input frames."""
    dag = decl["dags"][kind]
    return _value(dag["tensors"][decl["output"]]["symbolic_shape"]["dims"][int(decl["output_axis"])],
                  decl["symbol"], length)


def clip_output_frames(clip: Dict[str, Any], decl: Dict[str, Any], smap: Dict[str, int]) -> int:
    """The clip's output time extent at the request's binding."""
    node = clip["tensors"][decl["output"]]["symbolic_shape"]["dims"][int(decl["output_axis"])]
    return int(_symexpr.evaluate(node, lambda sid, _n: smap[sid], _symexpr.SHAPE))
