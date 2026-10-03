#!/usr/bin/env python3
"""Which dimensions did the trace leave at its own value? — a static scan of every container.

The owner's red line (2026-10-04): every model is traced with symbolic shapes, and a trace value
is never a size the model keeps. This reads each component's `graph.json` and lists every
dimension that is written as a plain integer where the SAME graph carries that number as a
symbolic expression — the signature of a dimension the tracer evaluated at its trace value
instead of carrying.

CONTRACT
--------
Input: a cache directory. A container is a sub-directory holding `manifest.json`; its graphs
are `components/*/graph.json`. Nothing is executed and nothing is written under the cache.

THE FIELD READ — named, as `docs/reference/a-frozen-dimension-is-read-from-the-arguments.md`
requires: `tensors[*].symbolic_shape.dims`, the tracer's expression annotation of each tensor
(the field the derived census keys from). `output_shapes` / `input_shapes` are a record of the
trace and are never read. For each hit the producer's `attributes.args` / `kwargs` are read too
and the record says whether the literal is ALSO in the arguments the runtime evaluates
(`in_args`) — mochi's rotary table carries 4 050 in its view's arguments; Qwen3-VL's expert view
carries `[128, -1, 2048]` and the 230 is in the annotation alone. Both are findings; the flag
tells the retrace which one it is fixing.

THE VALUES A FROZEN DIMENSION CAN TAKE (per component, from that component's own data):
  * `symbol`      — a declared symbol's trace value (`symbolic_context.symbols[*].trace_value`);
  * `expression`  — the `trace` of any expression node the graph carries anywhere (annotations
                    and arguments): `s0*s1`, `s1*((s2-2)//2+1)*((s3-2)//2+1)`, `s5+s1`. This is
                    every product, sum or floordiv of symbols the model itself computes — no
                    formula is enumerated here;
  * `trace-extent`— an extent of the trace inputs the topology recorded for the component
                    (`topology.json components[c].shapes`).
R39's collision set ({0, 1, 2}, `tools/symbol_collision_census.py`) is excluded: at those values
no rule is observable, so a literal there proves nothing either way. Two matches are WEAK — they
can make a hit AMBIGUOUS, never FROZEN: an expression that reaches its value only through a
symbol traced at a collision value (`s0*64` is 64 at batch 1, so every literal 64 would match),
and a topology extent of an input axis the graph keeps literal (fixed by design or never
symbolised: the graph alone cannot say which).

THE VALUES A MODEL LEGITIMATELY KEEPS (the AMBIGUOUS set): every extent of a WEIGHT (a parameter
that is not a registered buffer, and every shape in `weights_index.json`), every integer in the
topology's `extracted_values` for the component and `_global`, every integer in the component's
`profile.json` (its `config` and top-level values; not its memory, block or hint sections). A registered buffer is NOT in this set: a buffer allocated at the trace
length is frozen even when it is a model tensor.

CLASSES, per hit (a literal dim of a non-weight tensor equal to one of the first set):
  FROZEN     the value is not in the second set and its match is not weak;
  AMBIGUOUS  the value is in both (a model constant that coincides with a symbol's arithmetic, or
             a real freeze hidden behind one), or the match is weak; the record names the reason
             (`also`); read one by one, never counted as a defect;
and a component with no hit is CLEAN. A weight whose extent equals a trace value is not a hit:
that collision is the tracer's guard's business (and `symbol_collision_census.py`'s).

ORIGIN vs INHERITED: a hit whose producer consumes a tensor already carrying the same literal is
INHERITED (the freeze happened upstream); otherwise the producer is the hit's ORIGIN. Counts and
the "first op" name origins; inherited hits are counted beside them so the reach is visible.
A group is LIVE when one of its tensors is consumed or is a graph output (a norm's unused
mean/rstd output can carry a literal that no op ever reads). A hit is DROPPED when its producer
consumes a tensor carrying the same value symbolically and writes it back literal — the chain broke
at that very op, the strongest witness a static read gives.

OUTPUT: `--out` is a JSONL file, one record per container (counts, origin groups, the field read,
each graph's mtime), written after every container and rewritten sorted at the end; `<out stem>
_sites.jsonl` beside it holds every hit. Every exit — a refusal included — leaves a readable
`--out`: a refusal is a one-line record `{"refused": ...}` and a non-zero exit.

REFUSED BY NAME: a cache that does not exist, a cache holding no container, a container with no
component graph, a `--models` name the cache does not hold.

    python tools/frozen_dim_scan.py --out nbx/campaigns/<date>_frozen_scan/frozen_scan.jsonl
    python tools/frozen_dim_scan.py --cache DIR --models mochi-1-preview --out ...
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from symbol_collision_census import ARITHMETIC_COLLISIONS  # noqa: E402  (R39's set, one source)

FIELD = "tensors[*].symbolic_shape.dims"
FROZEN, AMBIGUOUS, CLEAN = "FROZEN", "AMBIGUOUS", "CLEAN"
_BINARY = {"mul": "*", "add": "+", "sub": "-", "floordiv": "//", "div": "/", "mod": "%", "pow": "**"}


class Refused(Exception):
    """An input the scan must not read as an answer — named, never a silent empty result."""


# ---------------------------------------------------------------- expressions

def expr_text(node) -> str:
    """A readable form of one dimension expression."""
    if isinstance(node, bool):
        return str(node)
    if isinstance(node, (int, float)):
        return str(node)
    if isinstance(node, dict):
        t = node.get("type")
        if t == "symbol":
            return str(_sid(node))
        if t in _BINARY:
            return f"({expr_text(node.get('left'))}{_BINARY[t]}{expr_text(node.get('right'))})"
        if t == "expression":
            f = node.get("factors") or node.get("terms") or []
            joiner = "*" if node.get("factors") else "+"
            return "(" + joiner.join(expr_text(x) for x in f) + ")"
        if t in ("max", "min", "neg", "ceildiv", "sym_max", "sym_min"):
            parts = [expr_text(node[k]) for k in ("left", "right", "operand", "value") if k in node]
            return f"{t}(" + ",".join(parts) + ")"
    return "?"


def expr_trace(node, symbols: dict):
    """The trace value of an expression node: its own `trace`, else evaluated from the symbol
    table. None when neither is possible (counted by the caller, never skipped silently)."""
    if isinstance(node, bool):
        return None
    if isinstance(node, int):
        return node
    if isinstance(node, str) and node in symbols:
        return symbols[node]
    if not isinstance(node, dict):
        return None
    for k in ("trace", "trace_value"):
        tr = node.get(k)
        if isinstance(tr, int) and not isinstance(tr, bool):
            return tr
    t = node.get("type")
    if t == "symbol":
        return symbols.get(_sid(node))
    if t in _BINARY:
        a, b = expr_trace(node.get("left"), symbols), expr_trace(node.get("right"), symbols)
        if a is None or b is None:
            return None
        try:
            return {"mul": a * b, "add": a + b, "sub": a - b, "floordiv": a // b if b else None,
                    "div": a // b if b and a % b == 0 else None, "mod": a % b if b else None,
                    "pow": a ** b}[t]
        except (TypeError, OverflowError):
            return None
    if t == "expression":
        vals = [expr_trace(x, symbols) for x in (node.get("factors") or node.get("terms") or [])]
        if not vals or any(v is None for v in vals):
            return None
        out = 1 if node.get("factors") else 0
        for v in vals:
            out = out * v if node.get("factors") else out + v
        return out
    return None


def _sid(node: dict):
    """A symbol node names its symbol as `id` (annotations) or `symbol_id` (some arguments, e.g.
    Sana-4K's text-encoder `arange`) — both encodings are in the catalogue."""
    v = node.get("id")
    return v if isinstance(v, str) else node.get("symbol_id")


def _is_expr(node) -> bool:
    return isinstance(node, dict) and (node.get("type") == "symbol" or node.get("type") in _BINARY
                                       or node.get("type") == "expression" or "trace" in node)


def _walk_exprs(node, out: list):
    """Every expression node (and sub-node) inside an annotation or an argument tree."""
    if isinstance(node, dict):
        if _is_expr(node):
            out.append(node)
        for v in node.values():
            _walk_exprs(v, out)
    elif isinstance(node, list):
        for v in node:
            _walk_exprs(v, out)


def _symbol_ids(node) -> set:
    out: set = set()
    if isinstance(node, dict):
        if node.get("type") == "symbol" and isinstance(_sid(node), str):
            out.add(_sid(node))
        for f in (node.get("factors") or []) + (node.get("terms") or []):
            if isinstance(f, str):
                out.add(f)
        for v in node.values():
            if isinstance(v, (dict, list)):
                out |= _symbol_ids(v)
    elif isinstance(node, list):
        for v in node:
            out |= _symbol_ids(v)
    return out


def _arg_literals(node, out: set):
    """Plain integers inside an op's args/kwargs — not inside an expression node (an expression's
    constant factor is part of a symbolic rule, not a frozen size)."""
    if isinstance(node, bool):
        return
    if isinstance(node, int):
        out.add(node)
    elif isinstance(node, dict):
        if _is_expr(node):
            return
        for k, v in node.items():
            if k not in ("trace", "trace_value"):
                _arg_literals(v, out)
    elif isinstance(node, list):
        for v in node:
            _arg_literals(v, out)


def _ints(node, out: set):
    if isinstance(node, bool):
        return
    if isinstance(node, int):
        out.add(node)
    elif isinstance(node, dict):
        for v in node.values():
            _ints(v, out)
    elif isinstance(node, list):
        for v in node:
            _ints(v, out)


# ---------------------------------------------------------------- one component

def tensor_kind(meta: dict, weight_names: set) -> str:
    """weight | buffer | constant | input | activation — from the graph's own flags."""
    if meta.get("is_parameter"):
        if meta.get("constant") and meta.get("weight_name") not in weight_names:
            return "buffer"
        return "weight"
    if meta.get("constant"):
        return "constant"
    if meta.get("is_input") or (str(meta.get("tensor_id") or "").startswith("input::")):
        return "input"
    return "activation"


def scan_graph(graph: dict, topology_shapes=None, model_constants=None, weight_index_shapes=None):
    """Scan one component graph. Returns (hits, info).

    `topology_shapes` — {input name: [extents]} the topology recorded for this component;
    `model_constants` — integers the model keeps (extracted values, profile config);
    `weight_index_shapes` — {weight name: shape} from weights_index.json.
    """
    syms_meta = (graph.get("symbolic_context") or {}).get("symbols") or {}
    symbols = {sid: m.get("trace_value") for sid, m in syms_meta.items()
               if isinstance(m, dict) and isinstance(m.get("trace_value"), int)}
    tensors = graph.get("tensors") or {}
    ops = graph.get("ops") or {}
    weight_index_shapes = weight_index_shapes or {}
    weight_names = set(weight_index_shapes)

    # -- the values a frozen dimension can take, each with the simplest expression naming it
    basis: dict = {}

    def offer(value, kind, text, weak=None):
        """`weak` names why a match on this value cannot by itself call a dimension frozen."""
        if not isinstance(value, int) or isinstance(value, bool) or value in ARITHMETIC_COLLISIONS or value < 0:
            return
        cur = basis.get(value)
        rank = (weak is not None, {"symbol": 0, "expression": 1, "trace-extent": 2}[kind], len(text))
        if cur is None or rank < cur[3]:
            basis[value] = (kind, text, weak, rank)

    for sid, v in symbols.items():
        offer(v, "symbol", f"{sid} ({syms_meta[sid].get('name')})")
    unevaluated = 0
    nodes: list = []
    for meta in tensors.values():
        _walk_exprs(((meta.get("symbolic_shape") or {}).get("dims")) or [], nodes)
    for op in ops.values():
        a = op.get("attributes") or {}
        _walk_exprs([a.get("args"), a.get("kwargs")], nodes)
    for node in nodes:
        if node.get("type") == "symbol":
            continue
        v = expr_trace(node, symbols)
        if v is None:
            unevaluated += 1
            continue
        ids = _symbol_ids(node)
        if not ids:
            continue                     # a constant wrapped as a node is not a symbol's value
        # A product through a symbol traced at an R39 collision value (batch at 1 or 2) equals
        # the other factor, or its double: `s0*64` is 64 at s0 = 1. A literal 64 can then be an
        # architecture constant or the product with the batch dropped — the trace cannot tell,
        # so the match is offered as AMBIGUOUS, never as FROZEN.
        colliding = any(symbols.get(i) in ARITHMETIC_COLLISIONS for i in ids)
        offer(v, "expression", expr_text(node),
              "matched only through a symbol at an R39 collision value" if colliding else None)
    # The topology's recorded trace extents. Where the graph symbolised the axis, the symbol
    # already names the value; what is left is an axis the graph keeps LITERAL — fixed by design
    # (a codec's hop, a channel count) or never symbolised — and the graph alone cannot say which
    # (VibeVoice's semantic tokenizer reads 3 200 samples = its hop at 24 kHz / 7.5 Hz). Offered
    # as AMBIGUOUS with that reason, never as FROZEN.
    for name, shp in (topology_shapes or {}).items():
        for i, d in enumerate(shp or []):
            offer(d, "trace-extent", f"topology {name}[{i}]",
                  "matched only by a trace extent of an input axis the graph keeps literal")

    # -- the values the model legitimately keeps
    keeps: dict = {}
    for tid, meta in tensors.items():
        if tensor_kind(meta, weight_names) == "weight":
            for d in meta.get("shape") or []:
                if isinstance(d, int):
                    keeps.setdefault(d, f"weight extent ({meta.get('weight_name') or tid})")
    for name, shp in weight_index_shapes.items():
        for d in shp or []:
            if isinstance(d, int):
                keeps.setdefault(d, f"weight extent ({name})")
    for d in (model_constants or {}):
        keeps.setdefault(d, f"model constant ({model_constants[d]})")

    # -- producer argument literals, read once
    arg_lits = {}
    for uid, op in ops.items():
        a = op.get("attributes") or {}
        s: set = set()
        _arg_literals([a.get("args"), a.get("kwargs")], s)
        arg_lits[uid] = s

    _traces_cache: dict = {}

    def symbolic_traces(tid):
        """The trace values a tensor's annotation carries SYMBOLICALLY."""
        if tid not in _traces_cache:
            dims = ((tensors.get(tid) or {}).get("symbolic_shape") or {}).get("dims") or []
            _traces_cache[tid] = {expr_trace(x, symbols) for x in dims if isinstance(x, dict)}
        return _traces_cache[tid]

    # -- hits, walked in execution order so an inherited literal finds its origin
    order = [u for u in (graph.get("execution_order") or []) if u in ops] or list(ops)
    origin_of: dict = {}            # (tensor id, value) -> origin op uid (or tensor id)
    outputs = set(graph.get("output_tensor_ids") or [])
    hits = []

    step = {uid: i for i, uid in enumerate(order)}

    def consider(tid, producer):
        meta = tensors.get(tid) or {}
        kind = tensor_kind(meta, weight_names)
        if kind == "weight":
            return
        dims = (meta.get("symbolic_shape") or {}).get("dims") or []
        for pos, d in enumerate(dims):
            if not isinstance(d, int) or isinstance(d, bool) or d not in basis:
                continue
            origin = None
            if producer is not None:
                for i in ops[producer].get("input_tensor_ids") or []:
                    if (i, d) in origin_of:
                        origin = origin_of[(i, d)]
                        break
            inherited = origin is not None
            origin = origin or producer or tid
            # DROPPED: the producer CONSUMES a tensor that carries this very value as a symbol or
            # an expression, and writes it back as a literal — the chain broke at this op. The
            # strongest evidence a static read can give; a literal born in a constructor's
            # arguments (`ones(23, 23)`) or matched by coincidence carries no such witness.
            dropped = bool(producer) and any(
                d in symbolic_traces(i) for i in ops[producer].get("input_tensor_ids") or [])
            origin_of[(tid, d)] = origin
            b = basis[d]
            also = keeps.get(d) or b[2]
            hits.append({
                "tensor": tid, "kind": kind, "position": pos, "value": d,
                "class": AMBIGUOUS if also else FROZEN,
                "matches": b[1], "basis": b[0], "also": also,
                "live": bool(meta.get("consumer_op_uids")) or tid in outputs,
                "producer": producer, "op_type": (ops.get(producer) or {}).get("op_type") if producer else None,
                "in_args": bool(producer and d in arg_lits.get(producer, ())),
                "dropped": dropped,
                "origin": origin, "inherited": inherited,
                "consumers": (meta.get("consumer_op_uids") or [])[:3],
                "step": step.get(producer, -1),
                "dims": [expr_text(x) if isinstance(x, dict) else x for x in dims],
            })

    for tid, meta in tensors.items():                       # sources first: inputs, buffers, constants
        if not meta.get("producer_op_uid"):
            consider(tid, None)
    for uid in order:
        for tid in ops[uid].get("output_tensor_ids") or []:
            consider(tid, uid)

    info = {"symbols": {sid: f"{(syms_meta[sid] or {}).get('name')}={v}" for sid, v in symbols.items()},
            "basis_values": len(basis), "unevaluated_expressions": unevaluated,
            "tensors": len(tensors), "ops": len(ops)}
    return hits, info


# ---------------------------------------------------------------- one container

def _component_inputs(container: Path):
    try:
        topo = json.loads((container / "topology.json").read_text())
    except (OSError, ValueError):
        return {}, {}, False
    comps = topo.get("components") or {}
    ev = topo.get("extracted_values") or {}
    return comps, ev, True


def _constants(ev_comp, ev_global, profile) -> dict:
    out = {}
    top = {k: v for k, v in (profile or {}).items() if k not in ("memory", "blocks", "globals", "hints", "version")}
    for label, src in (("extracted_values", ev_comp), ("extracted_values._global", ev_global),
                       ("profile", top)):
        s: set = set()
        _ints(src or {}, s)
        for v in s:
            out.setdefault(v, label)
    return out


def scan_container(container: Path) -> dict:
    graphs = sorted((container / "components").glob("*/graph.json"))
    rec = {"container": container.name, "field": FIELD, "components": {}, "verdict": None,
           "counts": {}, "patterns": [], "frozen": [], "ambiguous": []}
    if not graphs:
        rec["refused"] = f"{container.name}: no components/*/graph.json under {container}"
        return rec
    topo_comps, ev, topo_ok = _component_inputs(container)
    rec["topology_read"] = topo_ok
    sites = []
    for gp in graphs:
        comp = gp.parent.name
        try:
            graph = json.loads(gp.read_text())
        except (OSError, ValueError) as exc:
            rec["components"][comp] = {"unreadable": f"{type(exc).__name__}: {exc}"}
            continue
        try:
            profile = json.loads((gp.parent / "profile.json").read_text())
        except (OSError, ValueError):
            profile = {}
        try:
            wi = json.loads((gp.parent / "weights_index.json").read_text()).get("tensors") or {}
            wshapes = {k: v.get("shape") for k, v in wi.items() if isinstance(v, dict)}
        except (OSError, ValueError):
            wshapes = {}
        tshapes = ((topo_comps.get(comp) or {}).get("shapes")) or {}
        hits, info = scan_graph(graph, tshapes, _constants(ev.get(comp), ev.get("_global"), profile), wshapes)
        del graph
        info["graph_mtime"] = _dt.datetime.fromtimestamp(gp.stat().st_mtime).isoformat(timespec="seconds")
        c = collections.Counter((h["class"], "inherited" if h["inherited"] else "origin") for h in hits)
        info["counts"] = {f"{k[0]}_{k[1]}": n for k, n in sorted(c.items())}
        rec["components"][comp] = info
        for h in hits:
            h["component"] = comp
        sites.extend(hits)

    # origin groups: one row per (component, origin op, value) with its reach
    groups: dict = {}
    for h in sites:
        key = (h["component"], h["origin"], h["value"])
        g = groups.get(key)
        if g is None:
            g = groups[key] = {"component": h["component"], "origin": h["origin"], "value": h["value"],
                               "class": h["class"], "matches": h["matches"], "basis": h["basis"],
                               "also": h["also"], "kind": None, "op_type": None, "in_args": False,
                               "tensor": None, "position": None, "dims": None, "reach": 0,
                               "consumers": [], "inherited_tensors": [], "live": False, "dropped": False,
                               "step": h["step"]}
        g["reach"] += 1
        g["live"] = g["live"] or h["live"]
        g["dropped"] = g["dropped"] or h["dropped"]
        if not h["inherited"] and g["tensor"] is None:
            g.update(kind=h["kind"], op_type=h["op_type"], in_args=h["in_args"], tensor=h["tensor"],
                     position=h["position"], dims=h["dims"], consumers=h["consumers"])
        elif h["inherited"] and len(g["inherited_tensors"]) < 4:
            g["inherited_tensors"].append({"tensor": h["tensor"], "consumers": h["consumers"]})
    comp_rank = {c: i for i, c in enumerate(rec["components"])}
    rows = sorted(groups.values(), key=lambda g: (not g["live"], comp_rank.get(g["component"], 0), g["step"], g["origin"]))
    rec["frozen"] = [g for g in rows if g["class"] == FROZEN]
    rec["ambiguous"] = [g for g in rows if g["class"] == AMBIGUOUS]
    # PATTERNS: the same freeze repeated per block (48 expert views, 20 cross-attention slices) is
    # one pattern — (component, op type, position, value, match). What a retrace fixes is a pattern.
    pats: dict = {}
    for g in rec["frozen"]:
        k = (g["component"], g["op_type"], g["position"], g["value"], g["matches"])
        p = pats.get(k)
        if p is None:
            p = pats[k] = {"component": g["component"], "op_type": g["op_type"], "position": g["position"],
                           "value": g["value"], "matches": g["matches"], "kind": g["kind"],
                           "in_args": g["in_args"], "live": g["live"], "dropped": g["dropped"], "first": g["origin"],
                           "tensor": g["tensor"], "dims": g["dims"], "consumers": g["consumers"],
                           "origins": 0, "reach": 0}
        p["origins"] += 1
        p["reach"] += g["reach"]
        p["dropped"] = p["dropped"] or g["dropped"]
    rec["patterns"] = list(pats.values())
    rec["counts"] = {
        "FROZEN": len(rec["frozen"]), "AMBIGUOUS": len(rec["ambiguous"]),
        "FROZEN_live": sum(1 for g in rec["frozen"] if g["live"]),
        "FROZEN_patterns": len(rec["patterns"]),
        "FROZEN_patterns_live": sum(1 for p in rec["patterns"] if p["live"]),
        "FROZEN_sites": sum(1 for h in sites if h["class"] == FROZEN),
        "AMBIGUOUS_sites": sum(1 for h in sites if h["class"] == AMBIGUOUS),
        "components": len(graphs),
        "components_clean": sum(1 for c in rec["components"].values()
                                if not c.get("unreadable") and not c.get("counts")),
        "unreadable": sum(1 for c in rec["components"].values() if c.get("unreadable")),
    }
    rec["verdict"] = ("UNREADABLE" if rec["counts"]["unreadable"] else
                      FROZEN if rec["frozen"] else AMBIGUOUS if rec["ambiguous"] else CLEAN)
    rec["_sites"] = sites
    return rec


# ---------------------------------------------------------------- the catalogue

def containers_of(cache: Path, models=None) -> list:
    if not cache.is_dir():
        raise Refused(f"no cache at {cache}")
    found = sorted(p for p in cache.iterdir() if (p / "manifest.json").is_file())
    if not found:
        raise Refused(f"{cache} holds no container (no */manifest.json): a scan over zero "
                      f"containers is not a clean scan")
    if models:
        names = {p.name: p for p in found}
        missing = [m for m in models if m not in names]
        if missing:
            raise Refused(f"not in {cache}: {', '.join(missing)}")
        found = [names[m] for m in models]
    return found


def _first_frozen(rec: dict) -> str:
    if not rec.get("patterns"):
        return ""
    p = rec["patterns"][0]
    return f"{p['component']}/{p['first']} = {p['value']} ({p['matches']})"


def _write_lines(path: Path, recs: list):
    tmp = path.with_name(path.name + ".part")
    with tmp.open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def summary_table(recs: list) -> str:
    lines = [f"{'container':44s} {'verdict':10s} {'pat':>4s} {'live':>4s} {'FROZEN':>6s} {'AMBIG':>6s} {'f-sites':>7s} {'a-sites':>7s}  first FROZEN pattern"]
    for r in recs:
        if r.get("refused"):
            lines.append(f"{r['container']:44s} REFUSED    {r['refused']}")
            continue
        c = r["counts"]
        lines.append(f"{r['container']:44s} {r['verdict']:10s} {c['FROZEN_patterns']:4d} {c['FROZEN_patterns_live']:4d} {c['FROZEN']:6d} {c['AMBIGUOUS']:6d} "
                     f"{c['FROZEN_sites']:7d} {c['AMBIGUOUS_sites']:7d}  {_first_frozen(r)}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default=None, help="the cache root (default: the engine's own cache_dir())")
    ap.add_argument("--models", default=None, help="comma-separated container names; default every container")
    ap.add_argument("--out", required=True, help="JSONL, one record per container")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes (each holds one graph.json in memory)")
    ap.add_argument("--top", type=int, default=3, help="FROZEN origins printed per container")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sites_path = out.with_name(out.stem + "_sites.jsonl")
    if a.cache is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from neurobrix.core.paths import cache_dir
        cache = cache_dir()
    else:
        cache = Path(a.cache)
    started = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        conts = containers_of(cache, a.models.split(",") if a.models else None)
    except Refused as exc:
        _write_lines(out, [{"refused": str(exc), "cache": str(cache), "at": started}])
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    # largest first, so the long graphs do not end the run alone
    def weight(p):
        return -sum(g.stat().st_size for g in (p / "components").glob("*/graph.json"))
    conts.sort(key=weight)
    recs = []
    with sites_path.open("w") as sf:
        def take(rec):
            sites = rec.pop("_sites", [])
            for h in sites:
                sf.write(json.dumps({"container": rec["container"], **h}) + "\n")
            sf.flush()
            recs.append(rec)
            _write_lines(out, sorted(recs, key=lambda r: r["container"]))
            print(f"  {rec['container']:44s} {rec.get('verdict') or 'REFUSED':10s} "
                  f"{json.dumps(rec.get('counts'))}", flush=True)
        if a.jobs > 1:
            import multiprocessing as mp
            with ProcessPoolExecutor(max_workers=a.jobs, mp_context=mp.get_context("spawn")) as ex:
                for rec in ex.map(scan_container, conts):
                    take(rec)
        else:
            for p in conts:
                take(scan_container(p))
    recs.sort(key=lambda r: r["container"])
    _write_lines(out, recs)
    table = summary_table(recs)
    head = (f"frozen-dimension scan — field read: {FIELD} (never output_shapes)\n"
            f"cache {cache} · {len(recs)} container(s) · started {started} · finished "
            f"{_dt.datetime.now().astimezone().isoformat(timespec='seconds')}\n"
            f"pat = FROZEN patterns (component, op type, position, value, match); live = patterns with a tensor\n"
            f"that is consumed or is a graph output; FROZEN / AMBIG = origin groups (component, origin op, value);\n"
            f"f-/a-sites = every hit tensor\n")
    detail = []
    for r in recs:
        for g in (r.get("patterns") or [])[: a.top]:
            detail.append(f"  {r['container']}/{g['component']}  {g['first']} [{g['op_type']}] x{g['origins']} "
                          f"{g['tensor']}[{g['position']}] = {g['value']} ~ {g['matches']} "
                          f"({g['kind']}, in_args={g['in_args']}, dropped={g['dropped']}, live={g['live']}, reach {g['reach']}) "
                          f"dims={g['dims']} -> {g['consumers']}")
    text = head + "\n" + table + "\n\nfirst FROZEN origins per container:\n" + "\n".join(detail) + "\n"
    out.with_name(out.stem + "_summary.txt").write_text(text)
    print(text)
    refused = [r for r in recs if r.get("refused")]
    return 2 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
