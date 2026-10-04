"""An input reaches its graph at the extent the request gives it — never at the trace's.

A trace extent is a witnessed stimulus, not a value. Until 2026-10-04 the audio front ends
zero-padded or cut an input to the first stage's trace shape (`core/flow/rnnt.py`, its Triton mirror,
`audio_utils._fit_to_trace`, `audio_frontend.fit_features`, the audio flow's inline fit): every
clip ran at 3 000 mel frames (700 stacked frames for a conformer front end), so no graph ever met
another length, and a container whose graph froze that axis downstream (a pad mask repeated 375
times, a relative-position window sliced at 375) passed every gate. A runtime that pads to the
trace hides every length defect; the padding was the compensation of a build-side defect, and it
is removed here.

This module is the door that replaces it, shared by both engines' flows, by Prism's flow bindings
and by the derived census (one function, R30; torch-free, R33):

* `input_axes(dag)` — an input's trace shape and the symbol each of its axes carries, read from the
  graph's own symbol table (`symbolic_shape.dims`; the older encoding keeps its dicts in `shape`).
* `admit(container, component, dag, fed)` — the shapes a flow is about to feed:
    - an axis the graph carries as a SYMBOL takes the fed extent;
    - an axis the graph carries as a LITERAL takes its own extent only; any other is refused by
      name (`FrozenTraceExtent`: the container, the component, the input, the axis, the trace
      value, the extent fed, and the repair);
    - away from the trace point, the graph must FOLLOW the binding: every elementwise op whose
      inputs broadcast at the trace values must broadcast at the fed ones (`broadcast_breaks`).
      One that does not names a dim the container froze downstream of a symbolic input — refused by
      name with the op, the tensor, the dim and its trace value.

`broadcast_breaks` is the static scan of `tools/symbolic_broadcast_scan.py` evaluated at ONE
binding by the runtime's own resolver (`triton.symbols.SymbolResolver`); the tool calls it at its
own moved bindings. Nothing here knows a model, a family or an op's meaning beyond "its inputs
broadcast".

WHAT THE DOOR PROVES, AND WHAT IT DOES NOT. It refuses a frozen input axis and a frozen dim that
meets a symbolic one in an elementwise op — the class the two conformer containers carry. It is
not a proof that a graph is symbolic: a dim frozen in a view, an expand, a concatenation or a
contraction operand that never meets a symbolic neighbour in an elementwise op passes the door and
fails at its op. An op whose annotation the resolver cannot evaluate is refused too (the door does
not admit what it could not read). The symbols no fed input binds are held at their traced extent
for the scan: the question asked is "does the graph follow THIS feed", the other inputs unmoved.
"""
from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Tuple

#: Op types whose tensor inputs broadcast against each other (ATen's elementwise family).
BROADCASTING = frozenset({
    "aten::add", "aten::sub", "aten::mul", "aten::div", "aten::where", "aten::maximum",
    "aten::minimum", "aten::pow", "aten::eq", "aten::ne", "aten::lt", "aten::le", "aten::gt",
    "aten::ge", "aten::logical_and", "aten::logical_or", "aten::masked_fill", "aten::addcmul",
    "aten::addcdiv", "aten::lerp", "aten::atan2", "aten::remainder", "aten::fmod",
    "aten::bitwise_and", "aten::bitwise_or", "aten::copysign", "aten::hypot",
})

#: What a refusal tells its reader to do — the repair is Forge's, never a retry at another length.
REPAIR = ("rewrite it with the single-write pass (Forge's in-place re-propagation of the symbol "
          "table), or retrace it with that axis symbolic")


class FrozenTraceExtent(RuntimeError):
    """A container that cannot take the extent the request gives one of its inputs: the graph
    carries that axis — or a dim downstream of it — at its trace value. Refused by name."""


class InputAxes(NamedTuple):
    """One graph input: its tensor id, its name, its trace shape, {axis: symbol id} for the axes
    the graph carries as a bare symbol, and the axes it carries as a LITERAL (an axis in neither
    is an expression of symbols: not frozen, and bound by no single symbol)."""
    tensor_id: str
    name: str
    trace_shape: Tuple[int, ...]
    symbols: Dict[int, str]
    literal: Tuple[int, ...] = ()


def _dims(spec: dict) -> list:
    """An input's dims in the container's own encoding: `symbolic_shape.dims` when the container
    carries it, else `shape` (the older encoding keeps the symbolic dicts there)."""
    ss = spec.get("symbolic_shape")
    if isinstance(ss, dict) and isinstance(ss.get("dims"), list):
        return ss["dims"]
    return list(spec.get("shape") or [])


def _is_input(tid: str, spec: dict) -> bool:
    return (spec.get("type") == "input" or spec.get("input_name") is not None
            or tid.startswith("input::"))


def input_axes(dag: Optional[dict], name: Optional[str] = None) -> Optional[InputAxes]:
    """The axes of input `name` of a graph — its FIRST input when no name is given (the audio
    flows' convention: the first input is the features). None when the graph has no such input."""
    if not dag:
        return None
    tensors = dag.get("tensors") or {}
    ordered = [t for t in (dag.get("input_tensor_ids") or []) if t in tensors]
    ordered += [t for t in tensors if t not in ordered]
    for tid in ordered:
        spec = tensors[tid]
        if not _is_input(tid, spec):
            continue
        iname = spec.get("input_name") or (tid[len("input::"):] if tid.startswith("input::") else tid)
        if name is not None and iname != name:
            continue
        concrete = list(spec.get("shape") or [])
        trace, symbols, literal = [], {}, []
        for i, d in enumerate(_dims(spec)):
            if isinstance(d, dict):
                tv = d.get("trace_value", d.get("trace"))
                if tv is None and i < len(concrete) and isinstance(concrete[i], int):
                    tv = concrete[i]
                if not isinstance(tv, int):
                    raise FrozenTraceExtent(
                        f"input {iname!r} axis {i}: a symbolic dim with no trace value ({d!r}) — "
                        f"the container's symbol table is incomplete; {REPAIR}.")
                trace.append(tv)
                sid = d.get("id") or d.get("symbol_id")
                if d.get("type") == "symbol" and sid and not d.get("offset"):
                    symbols[i] = sid
            elif isinstance(d, int) and not isinstance(d, bool):
                trace.append(d)
                literal.append(i)
            else:
                raise FrozenTraceExtent(
                    f"input {iname!r} axis {i}: the container records neither an extent nor a "
                    f"symbol ({d!r}); {REPAIR}.")
        return InputAxes(tid, iname, tuple(trace), symbols, tuple(literal))
    return None


def features_and_length(dag: Optional[dict]) -> Tuple[str, str]:
    """(features input name, length input name) of a transducer encoder graph — read from the
    graph: its rank-3 input is the features, its rank-1 input their length."""
    feat = length = None
    tensors = (dag or {}).get("tensors") or {}
    for tid in (dag or {}).get("input_tensor_ids") or []:
        spec = tensors.get(tid) or {}
        name = spec.get("input_name") or tid.split("input::", 1)[-1]
        rank = len(spec.get("shape") or [])
        if rank == 3 and feat is None:
            feat = name
        elif rank == 1 and length is None:
            length = name
    if feat is None or length is None:
        raise RuntimeError(
            "ZERO FALLBACK: the encoder graph must take one rank-3 input (the features) and one "
            f"rank-1 input (their length) for the flow to read its input axes; found {feat!r} / "
            f"{length!r}.")
    return feat, length


class _Shape:
    """What the runtime's binder reads of a fed tensor: its shape."""

    def __init__(self, shape):
        self.shape = tuple(int(d) for d in shape)


def fed_bindings(dag: dict, fed: Dict[str, Any]) -> Dict[str, int]:
    """{symbol id: value} of a graph fed `fed` {input name: shape} — bound by the runtime's own
    binder; a symbol no fed input binds keeps its trace value (this feed does not move it)."""
    from neurobrix.triton.symbols import SymbolResolver
    ctx = dag.get("symbolic_context") or {}
    res = SymbolResolver(ctx)
    feed = {f"input::{k}": _Shape(v) for k, v in fed.items()}
    res.bind_from_inputs(feed, list(feed), dag.get("tensors") or {})
    out = dict(res.bindings)
    for sid, info in (ctx.get("symbols") or {}).items():
        if sid not in out and isinstance((info or {}).get("trace_value"), int):
            out[sid] = int(info["trace_value"])
    return out


def trace_bindings(dag: dict) -> Dict[str, int]:
    """{symbol id: the extent the graph was traced at}."""
    return {sid: int(info["trace_value"])
            for sid, info in ((dag.get("symbolic_context") or {}).get("symbols") or {}).items()
            if isinstance((info or {}).get("trace_value"), int)}


def _broadcasts(shapes: List[List[int]]) -> bool:
    rank = max(len(s) for s in shapes)
    for i in range(1, rank + 1):
        if len({s[-i] for s in shapes if len(s) >= i} - {1}) > 1:
            return False
    return True


def _resolver(dag: dict, bindings: Dict[str, int]):
    from neurobrix.triton.symbols import SymbolResolver
    res = SymbolResolver(dag.get("symbolic_context") or {})
    for sid, v in bindings.items():
        res._bind(sid, int(v))       # the resolver's single write site (its declared extents)
    return res


def broadcast_breaks(dag: dict, bindings: Dict[str, int],
                     trace: Optional[Dict[str, int]] = None) -> Tuple[List[dict], int]:
    """The elementwise ops whose tensor inputs broadcast at the trace values and NOT at `bindings`.

    Each input's `symbolic_shape.dims` is evaluated by the runtime's own resolver at both points;
    right-aligned dims must be equal or 1. Returns `(breaks, unevaluable)`: one record per breaking
    op — its uid, type and module, each input's tensor id with its dims at the trace and at the
    binding — and the number of ops carrying an expression the resolver cannot evaluate (counted,
    never judged). Read-only over the graph; nothing is run."""
    from neurobrix.triton.symbols import UnboundSymbolError
    trace = trace_bindings(dag) if trace is None else trace
    at_trace, at_fed = _resolver(dag, trace), _resolver(dag, bindings)
    tensors = dag.get("tensors") or {}
    found, unknown = [], 0
    for uid, op in (dag.get("ops") or {}).items():
        if op.get("op_type") not in BROADCASTING:
            continue
        dims = []
        for tid in op.get("input_tensor_ids") or []:
            ss = (tensors.get(tid) or {}).get("symbolic_shape")
            if isinstance(ss, dict) and isinstance(ss.get("dims"), list):
                dims.append((tid, ss["dims"]))
        if len(dims) < 2:
            continue
        try:
            a = [[at_trace.resolve(d) for d in ds] for _, ds in dims]
            b = [[at_fed.resolve(d) for d in ds] for _, ds in dims]
        except (UnboundSymbolError, TypeError, ValueError):
            unknown += 1
            continue
        if _broadcasts(a) and not _broadcasts(b):
            found.append({"op": uid, "op_type": op["op_type"], "module": op.get("parent_module"),
                          "inputs": [t for t, _ in dims], "dims": [ds for _, ds in dims],
                          "at_trace": a, "at_binding": b})
    return found, unknown


def _frozen_dim(brk: dict) -> str:
    """The dim a break names: on the first right-aligned axis where the inputs disagree at the
    binding, the input whose dim is a LITERAL is the one the container froze."""
    b, a = brk["at_binding"], brk["at_trace"]
    rank = max(len(s) for s in b)
    for i in range(1, rank + 1):
        if len({s[-i] for s in b if len(s) >= i} - {1}) <= 1:
            continue
        literal = [(tid, len(ds) - i, a[k][-i]) for k, (tid, ds) in
                   enumerate(zip(brk["inputs"], brk["dims"]))
                   if len(ds) >= i and not isinstance(ds[-i], dict) and a[k][-i] != 1]
        moved = [(tid, len(ds) - i, a[k][-i], b[k][-i]) for k, (tid, ds) in
                 enumerate(zip(brk["inputs"], brk["dims"]))
                 if len(ds) >= i and isinstance(ds[-i], dict)]
        if literal and moved:
            lt, la, lv = literal[0]
            mt, ma, mtr, mv = moved[0]
            return (f"{lt} dim {la} is the literal {lv} (its trace value) where {mt} dim {ma} "
                    f"follows the request ({mtr} at the trace, {mv} here)")
        vals = [(tid, s[-i]) for tid, s in zip(brk["inputs"], b) if len(s) >= i]
        return "the inputs disagree on a dim: " + ", ".join(f"{t} = {v}" for t, v in vals)
    return "the inputs do not broadcast"


def admit(container: Optional[str], component: str, dag: Optional[dict],
          fed: Dict[str, Any]) -> Dict[str, int]:
    """Admit the shapes a flow is about to feed `component`, or refuse by name.

    `fed` is {input name: shape}. Returns the bindings the feed implies. Raises
    `FrozenTraceExtent` when an input axis the graph froze is fed another extent, or when the
    graph does not follow the fed binding (`broadcast_breaks`). A graph fed at its trace point is
    admitted without the scan: the trace is the one point every container has witnessed."""
    who = f"container {container!r}, component {component!r}"
    if not dag:
        raise FrozenTraceExtent(f"{who}: no graph to read the input axes from.")
    for name, shape in fed.items():
        axes = input_axes(dag, name)
        if axes is None:
            raise FrozenTraceExtent(f"{who}: the graph has no input named {name!r}.")
        shape = tuple(int(d) for d in shape)
        if len(shape) != len(axes.trace_shape):
            raise FrozenTraceExtent(
                f"{who}: input {name!r} is fed rank {len(shape)} {shape}, the graph takes rank "
                f"{len(axes.trace_shape)} {axes.trace_shape}.")
        for axis, (got, trace) in enumerate(zip(shape, axes.trace_shape)):
            if axis not in axes.literal or got == trace:
                continue
            raise FrozenTraceExtent(
                f"{who}: input {name!r} axis {axis} is frozen at its trace extent {trace} — the "
                f"graph's symbol table carries no symbol for it — and the request gives it {got} "
                f"(fed shape {shape}, trace shape {axes.trace_shape}). The engine does not pad or "
                f"cut an input to a trace extent: {REPAIR}.")
    bindings = fed_bindings(dag, fed)
    trace = trace_bindings(dag)
    if all(bindings.get(s) == v for s, v in trace.items()):
        return bindings
    breaks, unknown = broadcast_breaks(dag, bindings, trace)
    if unknown:
        raise FrozenTraceExtent(
            f"{who}: {unknown} elementwise op(s) carry a shape expression the runtime's resolver "
            f"cannot evaluate, so the graph cannot be shown to follow the request's extent "
            f"(fed {dict(fed)}); the door does not admit what it could not read: {REPAIR}.")
    if breaks:
        first = breaks[0]
        names = (dag.get("symbolic_context") or {}).get("symbols") or {}
        moved = ", ".join(f"{s} ({(names.get(s) or {}).get('name', '?')}) {trace[s]} -> {bindings[s]}"
                          for s in sorted(trace) if bindings.get(s) != trace[s])
        raise FrozenTraceExtent(
            f"{who}: the graph does not follow the request's extent ({moved}). {len(breaks)} "
            f"elementwise op(s) broadcast at the trace and not here; the first is {first['op']} "
            f"({first['op_type']}, module {first['module']!r}): {_frozen_dim(first)} — shapes "
            f"{first['at_trace']} at the trace, {first['at_binding']} at the request. A dim "
            f"downstream of a symbolic input was frozen at its trace value; the engine does not "
            f"pad the input back to the trace to hide it: {REPAIR}.")
    return bindings


def fed_input(topology: Optional[dict], component: str, variable: str) -> Optional[str]:
    """The input of `component` the flow's `variable` reaches, by the topology's own connections
    (`global.input_features -> perception.audio_signal`); None when no connection names it."""
    short = variable.split(".")[-1]
    for conn in (topology or {}).get("connections") or []:
        dst = conn.get("to", "")
        if conn.get("from") in (variable, short, "global." + short) and dst.startswith(component + "."):
            return dst[len(component) + 1:]
    return None


def admit_features(container: Optional[str], topology: Optional[dict], component: Optional[str],
                   dag: Optional[dict], shape, variable: str) -> Optional[Dict[str, int]]:
    """Admit the features a front end produced as the input of the flow's first stage: the input
    the topology connects `variable` to, the graph's first input when no connection names it.
    The features are fed AS PRODUCED — the front end's own contract decides their extent (the
    whisper extractor's 30 s window is the vendor's; a NeMo or conformer front end yields the
    recording's own frames) — and the graph either carries that extent or is refused by name.
    A first stage that has NO GRAPH in this run (no executor was built for it) cannot be judged
    and is not run by the flow either: None, and the features stay bound for whoever reads them."""
    if component is None or not dag:
        return None
    axes = input_axes(dag, fed_input(topology, component, variable))
    if axes is None:
        raise FrozenTraceExtent(
            f"container {container!r}, component {component!r}: the graph has no input for the "
            f"flow's {variable!r}.")
    return admit(container, component, dag, {axes.name: tuple(shape)})
