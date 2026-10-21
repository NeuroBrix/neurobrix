"""Split a stretch of a component's graph along one of its token axes, at the source.

WHY THIS EXISTS

Layer streaming cuts a component BETWEEN ops and reserves the activations' peak in every piece. When
the activations of a single stretch of ops — one attention, one feed-forward — are over the budget
on their own, no cut between ops can help: the refusal reads "activations alone peak at ...", and
the guidance split (one branch at a time) is the last lever the flow holds. Measured 2026-10-05,
SANA-Video_2B_720p on the 16 GB V100 at the 8 192 MB rung, ONE branch: the transformer peaks at
9 486.7 MB after `aten.mul::22` (ReLU linear attention + rotary, every tensor [1, 155 232, ...]) and
9 420.4 MB at the feed-forward's 13 440-channel expand.

Those ops are PER TOKEN: each output row depends on its own input row. Run on a slice of the token
axis, they give that slice of the output. So the stretch is run K times on 1/K of the axis, and its
activations fall by K. What is not per token is classified exactly, and decides where the stretch
must end — or how it is staged:

  * a CONTRACTION over the axis (a `sum` over it, a product whose inner dimension is it) is
    additive: each slice contributes a partial, and the partials are summed. Linear attention's
    k^T v and its normaliser are two. Their consumers need the WHOLE sum, so they run in a later
    PASS over the slices, recomputing the per-token tensors they read (never storing them whole).
  * a MIXING op (a softmax over the axis, an attention whose keys carry it, a temporal convolution
    of kernel > 1, a cumsum, an `arange` of it) cannot run on a slice: the stretch ends before it.

HOW THE AXIS IS FOUND — FROM DATAFLOW, NEVER FROM A NAME

A token axis is a SYMBOL (`s6`, frames) bound from a component input's dimension. A tensor carries
it when its symbolic shape mentions it; the dimension that does is a PRODUCT `P * S * R` — S's
extent between an outer extent P and an inner extent R, in row-major order. Forge's symbolic
algebra keeps the vendor's arithmetic order (`SymInt.__mul__` does not sort), so the textual order
of `((s6*s7)*s8)` is NOT the memory order: the layout is DERIVED, op by op from the tensors where
the dimension IS the symbol (R = 1), through every reshape, by exact monomial arithmetic. A slice of
S is then `x.view(.., P, S, R, ..).narrow(S-axis, f0, c)`, the same in both engines.

Pure Python: imported by Prism (pricing) and by the streaming strategy (execution) in every engine
(R33). It names no model, no family, no layer.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

FREE, SLICE, CONTRACT, MIX = "free", "slice", "contract", "mix"


# ---------------------------------------------------------------------------------------------
# Monomials: an integer coefficient times atoms (a symbol id, or an opaque expression without S)
# ---------------------------------------------------------------------------------------------

def _strip_trace(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _strip_trace(v) for k, v in node.items() if k != "trace"}
    if isinstance(node, list):
        return [_strip_trace(v) for v in node]
    return node


def _canon(node: Any) -> str:
    return json.dumps(_strip_trace(node), sort_keys=True, separators=(",", ":"))


def _symbol_id(node: Any) -> Optional[str]:
    if isinstance(node, dict) and node.get("type") == "symbol":
        sid = node.get("id") or node.get("symbol_id")
        return str(sid) if sid is not None else None
    return None


def mentions(node: Any, sym: str) -> bool:
    """Whether an expression (or a list of them) reads the symbol `sym`, in every form the
    resolvers accept: a symbol node, a `symbol_id` key, a bare string that IS the id."""
    if isinstance(node, dict):
        if node.get("type") == "symbol" and (node.get("id") == sym or node.get("symbol_id") == sym):
            return True
        if node.get("symbol_id") == sym:
            return True
        return any(mentions(v, sym) for k, v in node.items() if k != "trace")
    if isinstance(node, (list, tuple)):
        return any(mentions(v, sym) for v in node)
    return isinstance(node, str) and node == sym


class Mono:
    """coef * prod(atom ** exp). An atom is `sym:<id>` or `expr:<canonical json>`."""

    __slots__ = ("coef", "atoms")

    def __init__(self, coef: int = 1, atoms: Optional[Dict[str, int]] = None):
        self.coef = int(coef)
        self.atoms: Tuple[Tuple[str, int], ...] = tuple(sorted((k, v) for k, v in (atoms or {}).items() if v))

    def _d(self) -> Dict[str, int]:
        return dict(self.atoms)

    def __mul__(self, other: "Mono") -> "Mono":
        d = self._d()
        for k, v in other.atoms:
            d[k] = d.get(k, 0) + v
        return Mono(self.coef * other.coef, d)

    def div(self, other: "Mono") -> Optional["Mono"]:
        """self / other when exact, else None."""
        if other.coef == 0 or self.coef % other.coef:
            return None
        d = self._d()
        for k, v in other.atoms:
            if d.get(k, 0) < v:
                return None
            d[k] -= v
        return Mono(self.coef // other.coef, d)

    def deg(self, sym: str) -> int:
        return dict(self.atoms).get("sym:" + sym, 0)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mono) and self.coef == other.coef and self.atoms == other.atoms

    def __hash__(self) -> int:
        return hash((self.coef, self.atoms))

    def is_one(self) -> bool:
        return self.coef == 1 and not self.atoms

    def evaluate(self, symbol: Callable[[str], int]) -> int:
        from neurobrix.core.runtime.symexpr import SHAPE, evaluate
        out = self.coef
        for key, exp in self.atoms:
            if key.startswith("sym:"):
                val = int(symbol(key[4:]))
            else:
                val = int(evaluate(json.loads(key[5:]), lambda sid, _n: symbol(str(sid)), SHAPE))
            out *= val ** exp
        return out

    def __repr__(self) -> str:
        parts = [str(self.coef)] if self.coef != 1 or not self.atoms else []
        for k, v in self.atoms:
            name = k[4:] if k.startswith("sym:") else "(" + k[5:][:30] + ")"
            parts.append(name if v == 1 else f"{name}^{v}")
        return "*".join(parts)


ONE = Mono(1)


def mono_of(node: Any, sym: str) -> Optional[Mono]:
    """The monomial of one dimension, or None when it reads `sym` other than as a factor (an
    offset, a sum, a floor division) — a dimension that does is not a slice of the axis."""
    if isinstance(node, bool):
        return None
    if isinstance(node, int):
        return Mono(node)
    if isinstance(node, str):
        return Mono(1, {"sym:" + node: 1})
    if not isinstance(node, dict):
        return None
    t = node.get("type")
    sid = _symbol_id(node)
    if sid is not None and not node.get("offset"):
        return Mono(1, {"sym:" + sid: 1})
    if t == "mul":
        a, b = mono_of(node.get("left"), sym), mono_of(node.get("right"), sym)
        return a * b if a is not None and b is not None else None
    if t == "product":
        out = ONE
        for f in node.get("factors") or []:
            m = mono_of(f, sym)
            if m is None:
                return None
            out = out * m
        return out
    if t in ("const", "scalar") or (t is None and "value" in node):
        v = node.get("value")
        return Mono(v) if isinstance(v, int) and not isinstance(v, bool) else None
    if mentions(node, sym):
        return None
    return Mono(1, {"expr:" + _canon(node): 1})


def _poly_of(node: Any, units: Dict[str, int]) -> Optional[Dict[Tuple[Tuple[str, int], ...], int]]:
    """The dimension as a polynomial with each symbol t of `units` replaced by `units[t] * T` (T keeps the
    key `sym:<t>`; a symbol `units` does not name is itself):
    {atoms: integer coefficient}, atoms a sorted tuple of (key, exponent), () the constant. Python's
    arithmetic, read exactly: a floor division or a modulo by an integer m resolves only when m divides
    every coefficient but the constant's — then floor((m*X + c) / m) = X + floor(c / m) and
    (m*X + c) % m = c % m for every integer X. None for anything else (a real, a division by an
    expression, a remainder that depends on U)."""
    def const(c: int) -> Dict:
        return {(): int(c)} if c else {}

    def add(x: Dict, y: Dict, sign: int = 1) -> Dict:
        out = dict(x)
        for k, v in y.items():
            out[k] = out.get(k, 0) + sign * v
            if not out[k]:
                del out[k]
        return out

    def mul(x: Dict, y: Dict) -> Dict:
        out: Dict = {}
        for ka, va in x.items():
            for kb, vb in y.items():
                d = dict(ka)
                for key, e in kb:
                    d[key] = d.get(key, 0) + e
                k = tuple(sorted(d.items()))
                out[k] = out.get(k, 0) + va * vb
                if not out[k]:
                    del out[k]
        return out

    def divisor(x: Optional[Dict]) -> Optional[int]:
        if x is None or any(k for k in x) or not x:
            return None
        return int(x[()])

    def walk(n: Any) -> Optional[Dict]:
        if isinstance(n, bool):
            return None
        if isinstance(n, int):
            return const(n)
        if isinstance(n, str):
            return {(("sym:" + n, 1),): units.get(n, 1)}
        if not isinstance(n, dict):
            return None
        t = n.get("type")
        sid = _symbol_id(n)
        if sid is not None:
            base = {(("sym:" + sid, 1),): units.get(sid, 1)}
            off = n.get("offset", 0)
            return add(base, const(off)) if isinstance(off, int) else None
        if t in ("const", "scalar") or (t is None and "value" in n):
            v = n.get("value")
            return const(v) if isinstance(v, int) and not isinstance(v, bool) else None
        if not any(mentions(n, t) for t in units):
            return {(("expr:" + _canon(n), 1),): 1}
        if t == "neg":
            x = walk(n.get("operand", n.get("left")))
            return None if x is None else {k: -v for k, v in x.items()}
        if t == "product":
            out = const(1)
            for f in n.get("factors") or []:
                x = walk(f)
                if x is None:
                    return None
                out = mul(out, x)
            return out
        if t in ("add", "sub", "mul", "floordiv", "mod"):
            x, y = walk(n.get("left")), walk(n.get("right"))
            if x is None or y is None:
                return None
            if t == "add":
                return add(x, y)
            if t == "sub":
                return add(x, y, -1)
            if t == "mul":
                return mul(x, y)
            m = divisor(y)
            if not m or any(v % m for k, v in x.items() if k):
                return None
            c = x.get((), 0)
            if t == "mod":
                return const(c % m)
            return add({k: v // m for k, v in x.items() if k}, const(c // m))
        return None

    return walk(node)


def mono_of_unit(node: Any, sym: str, units: Optional[Dict[str, int]]) -> Optional[Mono]:
    """`mono_of` with the symbols of `units` read in their units (`_poly_of`): the one term the dimension's
    polynomial must be, or None. No units: `mono_of`."""
    if not units:
        return mono_of(node, sym)
    poly = _poly_of(node, units)
    if poly is None or len(poly) != 1:
        return None
    (atoms, coef), = poly.items()
    return Mono(coef, dict(atoms)) if coef else None


def symbol_units(graph: Dict[str, Any]) -> Dict[str, int]:
    """{symbol: the least common multiple of every integer the graph floor-divides or reduces an expression of
    it by} — the units every symbol is read in when one axis is sliced in units (`TokenAxis`): HAT's height and
    width both come in windows of 16, and the inner extent of its window view is s2 // 16 * 16, which only a
    reading of s2 in its own units relates to s2. A reading in units holds only where the symbol's value is a
    multiple of its unit: the planner checks it at the request, the runtime at every run (`TokenAxis.scaled`)."""
    syms = ((graph.get("symbolic_context") or {}).get("symbols") or {})
    out: Dict[str, int] = {}
    for sid in syms:
        got = axis_units(graph, sid)
        if got:
            out[sid] = got[0]
    return out


def axis_units(graph: Dict[str, Any], sym: str) -> List[int]:
    """The units an axis can be sliced in beyond 1: the least common multiple of every integer the graph's
    extents and size lists floor-divide or reduce an expression of `sym` by (HAT's window partition views
    its height `s1` as (s1 // 16, 16): its rows come in windows of 16, and a slice of whole windows keeps
    every such view exact). [] when the graph divides `sym` by nothing."""
    found: Set[int] = set()

    def walk(n: Any) -> None:
        if isinstance(n, dict):
            if n.get("type") in ("floordiv", "mod") and isinstance(n.get("right"), int) \
                    and not isinstance(n.get("right"), bool) and n["right"] > 1 and mentions(n.get("left"), sym):
                found.add(int(n["right"]))
            for k, v in n.items():
                if k != "trace":
                    walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    for t in (graph.get("tensors") or {}).values():
        walk((t or {}).get("symbolic_shape"))
    for op in (graph.get("ops") or {}).values():
        walk(((op or {}).get("attributes") or {}).get("args"))
    if not found:
        return []
    return [math.lcm(*sorted(found))]


# ---------------------------------------------------------------------------------------------
# Layouts and verdicts
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Layout:
    """Where S lives in a tensor: dimension `axis`, whose extent is `dim` = P * S * R. `inner` is R;
    None marks a BROADCAST along the axis (expanded from 1): constant along it, any slice of it is
    every slice."""
    axis: int
    dim: Mono
    inner: Optional[Mono]


@dataclass
class Verdict:
    kind: str
    reason: str = ""
    #: A local op whose output row reads its input rows within this radius (a same-padded convolution's
    #: dilation * (kernel - 1) / 2), in rows of its INPUT along the axis. A stretch holding such ops runs on
    #: slices widened by their accumulated receptive field and keeps each slice's zone of responsibility
    #: (NIST, exact tile-based inference: halo = half the receptive field) — `RegionPlan.margin`.
    halo: int = 0
    #: A cyclic shift along the axis (`aten.roll`): its halo is read ROUND the axis' ends — a slice at either end
    #: reads the rows the roll brings in from the other end, never padding (`RegionPlan.cyclic`, `slice_reads`).
    cyclic: bool = False


_POINTWISE = frozenset("""
add sub mul div rsub relu gelu silu sigmoid tanh exp exp2 log log2 log1p expm1 sqrt rsqrt reciprocal
pow neg abs clamp clamp_min clamp_max where sin cos tan erf erfc sign floor ceil round trunc frac
eq ne lt le gt ge logical_not logical_and logical_or logical_xor bitwise_not bitwise_and bitwise_or
minimum maximum fmod remainder masked_fill hardtanh hardswish hardsigmoid leaky_relu elu softplus mish
square _to_copy clone contiguous alias detach lift_fresh_copy ones_like zeros_like full_like empty_like
rand_like_never fill nan_to_num isnan isinf isfinite custom::swiglu_fused
""".split()) - {"rand_like_never"}

#: Ops that only move or keep the axis: every S-carrying output's S dimension is the input's.
_AXIS_MOVING = frozenset("permute transpose t unsqueeze squeeze alias detach view_as_real".split())

#: Reductions and scans whose `dim` argument decides: over the axis they mix, elsewhere they carry.
_DIM_REDUCE = frozenset("""
mean amax amin max min argmax argmin var std var_mean std_mean prod any all logsumexp
cumsum cumprod softmax _softmax log_softmax _log_softmax flip
""".split())

#: Ops selecting along `dim`: along any other dimension they carry the axis.
_DIM_SELECT = frozenset("split split_with_sizes chunk unbind slice select narrow index_select".split())

_VIEWS = frozenset("view _unsafe_view reshape flatten unflatten".split())

_FACTORIES = frozenset("zeros ones full empty new_zeros new_ones new_full new_empty empty_strided".split())

#: Attention ops: (position of is_causal, position of dropout_p) in aten's signatures.
_SDPA_FLAGS = {
    "_scaled_dot_product_efficient_attention": (6, 5),   # q k v attn_bias compute_lse dropout_p is_causal
    "_scaled_dot_product_flash_attention": (4, 3),       # q k v dropout_p is_causal ...
    "scaled_dot_product_attention": (5, 4),              # q k v attn_mask dropout_p is_causal
    "_scaled_dot_product_flash_attention_for_cpu": (4, 3),
    "_scaled_dot_product_cudnn_attention": (6, 5),
}
_SDPA = frozenset(_SDPA_FLAGS)


def _short(op_type: str) -> str:
    return op_type[6:] if op_type.startswith("aten::") else op_type


def _raw(arg: Any) -> Any:
    """An op argument as a plain value: a tensor as ('T', tid), a list as a list of raw items."""
    if isinstance(arg, dict):
        ty = arg.get("type")
        if ty in ("tensor", "tensor_ref"):
            return ("T", arg.get("tensor_id"))
        if ty == "tensor_tuple":
            return [("T", t) for t in arg.get("tensor_ids") or []]
        if ty == "list":
            return [_raw(v) for v in arg.get("value") or []]
        if ty == "scalar":
            return arg.get("value")
        return arg
    return arg


def _norm_dim(d: Any, rank: int) -> Optional[int]:
    if isinstance(d, bool) or not isinstance(d, int):
        return None
    return d % rank if rank else None


class TokenAxis:
    """The layout of one symbol across a whole graph, and every op's verdict for it."""

    def __init__(self, graph: Dict[str, Any], sym: str, unit: int = 1):
        self.graph = graph
        self.sym = sym
        #: The axis is sliced in units of `unit` of the symbol (`axis_units`): S stands for symbol / unit, and
        #: every symbol the graph divides is read in its own unit (`symbol_units`, `mono_of_unit`) so a view's
        #: extents relate exactly. 1: the symbols themselves, no reading in units.
        self.unit = int(unit)
        self.units: Dict[str, int] = {}
        if self.unit > 1:
            self.units = dict(symbol_units(graph))
            self.units[sym] = self.unit
        self.S = Mono(1, {"sym:" + sym: 1})
        self.tensors: Dict[str, Any] = graph.get("tensors") or {}
        self.ops: Dict[str, Any] = graph.get("ops") or {}
        self.order: List[str] = list(graph.get("execution_order") or [])
        self.index = {u: i for i, u in enumerate(self.order)}
        self.readers: Dict[str, List[int]] = {}
        #: tensor -> index of the op producing it, from the ops themselves (a fusion leaves a
        #: tensor's recorded `producer_op_uid` naming an op no longer in the order).
        self.producer: Dict[str, int] = {}
        for i, uid in enumerate(self.order):
            op = self.ops.get(uid) or {}
            for tid in op.get("input_tensor_ids") or []:
                self.readers.setdefault(tid, []).append(i)
            for tid in op.get("output_tensor_ids") or []:
                self.producer.setdefault(tid, i)
        self.graph_outputs = set(graph.get("output_tensor_ids") or [])
        self.layout: Dict[str, Layout] = {}
        #: S-carrying tensors without a derivable layout, with why.
        self.bad: Dict[str, str] = {}
        self.verdicts: List[Verdict] = []
        self._dims_memo: Dict[str, Optional[List[Optional[Mono]]]] = {}
        self._propagate()

    # -- tensor facts ----------------------------------------------------------------------------

    def dims(self, tid: str) -> List[Any]:
        t = self.tensors.get(tid) or {}
        ss = t.get("symbolic_shape")
        if isinstance(ss, dict) and isinstance(ss.get("dims"), list):
            return list(ss["dims"])
        return list(t.get("shape") or [])

    def carries(self, tid: str) -> bool:
        return mentions(self.dims(tid), self.sym)

    def monos(self, tid: str) -> Optional[List[Optional[Mono]]]:
        if tid not in self._dims_memo:
            self._dims_memo[tid] = [mono_of_unit(d, self.sym, self.units) for d in self.dims(tid)]
        return self._dims_memo[tid]

    def s_dim(self, tid: str) -> Tuple[Optional[int], Optional[Mono], str]:
        """(axis, monomial) of the one dimension carrying S, or (None, None, why)."""
        ds = self.dims(tid)
        hits = [i for i, d in enumerate(ds) if mentions(d, self.sym)]
        if len(hits) != 1:
            return None, None, f"S in {len(hits)} dimensions of {tid}"
        m = self.monos(tid)[hits[0]]
        if m is None or m.deg(self.sym) != 1:
            return None, None, f"{tid} dimension {hits[0]} reads S other than as one factor"
        return hits[0], m, ""

    def _source(self, tid: str) -> Optional[Layout]:
        axis, m, _ = self.s_dim(tid)
        if axis is not None and m == self.S:
            return Layout(axis, m, ONE)
        if axis is not None and self.unit > 1 and m == Mono(self.unit) * self.S:
            # the symbol's own rows, in units of `unit`: row r is unit r // unit, the remainder inner
            return Layout(axis, m, Mono(self.unit))
        return None

    def scaled(self, symbol: Callable[[str], int]) -> Callable[[str], int]:
        """The reader a monomial of this axis is evaluated with: each symbol in its unit (`self.units`), refused by
        name where its value is not a whole number of them — the reading in units would not be the graph's."""
        def value(sid: str) -> int:
            v, unit = int(symbol(sid)), self.units.get(sid, 1)
            q, r = divmod(v, unit)
            if r:
                raise RuntimeError(f"chunked region: symbol {sid} = {v} is not a whole number of its units of {unit} "
                                   f"(the graph's views of it in units would not hold)")
            return q
        return value

    def fits_units(self, symbol_map: Dict[str, int]) -> bool:
        """Whether every symbol this axis reads in units is a whole number of them at `symbol_map`."""
        return all(int(symbol_map.get(t, 0)) % u == 0 for t, u in self.units.items() if t in symbol_map)

    def p_of(self, lay: Layout) -> Optional[Mono]:
        if lay.inner is None:
            return None
        return lay.dim.div(self.S * lay.inner)

    # -- propagation -----------------------------------------------------------------------------

    def _lay(self, tid: str) -> Optional[Layout]:
        if tid in self.layout:
            return self.layout[tid]
        if tid in self.bad:
            return None
        t = self.tensors.get(tid) or {}
        if tid in self.producer:
            return None             # produced by an op that gave it no layout (set in _propagate)
        if t.get("is_parameter") or tid.startswith(("param::", "buffer::")):
            self.bad[tid] = f"weight {tid} carries S"
            return None
        lay = self._source(tid)
        if lay is None:
            self.bad[tid] = f"input {tid}: its S dimension is not S itself, so its layout is unknown"
            return None
        self.layout[tid] = lay
        return lay

    def _propagate(self) -> None:
        for i, uid in enumerate(self.order):
            op = self.ops.get(uid) or {}
            outs = list(op.get("output_tensor_ids") or [])
            try:
                v, lays = self._rule(op)
            except _Mix as exc:
                v, lays = Verdict(MIX, str(exc)), {}
            if v.kind == SLICE:
                lost = [t for t in outs if t not in lays and not self.carries(t)
                        and (self.readers.get(t) or t in self.graph_outputs)]
                if lost:
                    v, lays = Verdict(MIX, f"output {lost[0]} loses the axis and is read"), {}
            self.verdicts.append(v)
            for t in outs:
                if not self.carries(t):
                    continue
                if v.kind in (SLICE, FREE) and t in lays:
                    self.layout[t] = lays[t]
                    continue
                src = self._source(t)
                if src is not None:
                    self.layout[t] = src           # a mixing op's output starts a fresh axis
                else:
                    self.bad[t] = f"produced by {uid} ({v.reason or v.kind})"

    # -- per-op rules ----------------------------------------------------------------------------

    def _carry(self, out: str, lay: Layout) -> Layout:
        """An output whose S dimension IS the input's: same extent, same inner, wherever it sits."""
        axis, m, why = self.s_dim(out)
        if axis is None:
            raise _Mix(why)
        if m != lay.dim:
            raise _Mix(f"{out}'s S dimension {m} is not its input's {lay.dim}")
        return Layout(axis, m, lay.inner)

    def _no_s_args(self, op: Dict[str, Any], allow_lists: bool = False) -> None:
        for a in (op.get("attributes") or {}).get("args") or []:
            r = _raw(a)
            if isinstance(r, tuple) or (isinstance(r, list) and all(isinstance(x, tuple) for x in r)):
                continue
            if isinstance(r, list) and allow_lists:
                continue
            if mentions(r, self.sym):
                raise _Mix("an argument depends on S")

    def _args(self, op: Dict[str, Any]) -> List[Any]:
        return [_raw(a) for a in (op.get("attributes") or {}).get("args") or []]

    def _rule(self, op: Dict[str, Any]) -> Tuple[Verdict, Dict[str, Layout]]:
        ty = _short(str(op.get("op_type") or ""))
        ins = [t for t in (op.get("input_tensor_ids") or [])]
        outs = list(op.get("output_tensor_ids") or [])
        s_in = [t for t in ins if self.carries(t)]
        s_out = [t for t in outs if self.carries(t)]
        for t in s_in:
            if self._lay(t) is None:
                raise _Mix(self.bad.get(t) or f"{t} has no layout")
        if not s_in and not s_out:
            self._no_s_args(op)
            return Verdict(FREE), {}
        if not s_in:
            return self._rule_born(ty, op, s_out)
        lays = {t: self.layout[t] for t in s_in}
        if ty in _POINTWISE or ty == "custom::rms_norm" or ty == "native_layer_norm" \
                or ty == "_native_batch_norm_legit_no_training":
            return self._rule_pointwise(ty, op, ins, outs, lays)
        if ty in _AXIS_MOVING or ty in ("clone", "contiguous", "_to_copy", "view_as_complex"):
            self._no_s_args(op)
            lay = lays[s_in[0]]
            if ty == "view_as_complex" and lay.axis == len(self.dims(s_in[0])) - 1:
                raise _Mix("view_as_complex over the axis")
            return Verdict(SLICE), {t: self._carry(t, lay) for t in s_out}
        if ty == "expand":
            return self._rule_expand(op, s_in[0], lays[s_in[0]], outs)
        if ty in _VIEWS:
            return self._rule_view(ty, op, s_in[0], lays[s_in[0]], outs)
        if ty in _DIM_SELECT or ty in _DIM_REDUCE or ty == "sum":
            return self._rule_dim(ty, op, ins, s_in, lays, outs)
        if ty in ("cat", "stack"):
            return self._rule_cat(ty, op, s_in, lays, outs)
        if ty in ("mm", "bmm", "addmm", "baddbmm", "linear"):
            return self._rule_matmul(ty, op, ins, lays, outs)
        if ty == "embedding":
            self._no_s_args(op)
            if ins and ins[0] in s_in:
                raise _Mix("the embedding table carries S")
            return Verdict(SLICE), {t: self._carry(t, lays[s_in[0]]) for t in s_out}
        if ty in ("constant_pad_nd", "reflection_pad1d", "reflection_pad2d", "reflection_pad3d",
                  "replication_pad1d", "replication_pad2d", "replication_pad3d"):
            return self._rule_pad(op, s_in[0], lays[s_in[0]], outs)
        if ty == "convolution":
            return self._rule_conv(op, ins, s_in, lays, outs)
        if ty == "im2col":
            return self._rule_im2col(op, ins, s_in, lays, outs)
        if ty == "roll":
            return self._rule_roll(op, s_in[0], lays[s_in[0]], outs)
        if ty in ("upsample_nearest2d", "upsample_nearest1d", "upsample_nearest3d", "pixel_shuffle"):
            return self._rule_scale(ty, op, s_in[0], lays[s_in[0]], outs)
        if ty == "native_group_norm":
            lay = lays[s_in[0]]
            if s_in != ins[:1] or lay.axis != 0:
                raise _Mix("group norm over the axis")
            return Verdict(SLICE), {t: self._carry(t, lay) for t in s_out}
        if ty in _SDPA:
            return self._rule_sdpa(op, ins, s_in, lays, outs)
        raise _Mix(f"{ty} has no token-axis rule")

    def _rule_born(self, ty: str, op: Dict[str, Any], s_out: List[str]) -> Tuple[Verdict, Dict[str, Layout]]:
        """An op with no S input producing S: a broadcast is a wildcard; anything else (arange,
        linspace, a view of a frozen tensor) mints positions and mixes."""
        if ty == "expand":
            src = (op.get("input_tensor_ids") or [None])[0]
            ods = self.dims(s_out[0])
            ids = self.dims(src) if src else []
            axis, m, why = self.s_dim(s_out[0])
            if axis is None:
                raise _Mix(why)
            j = axis - (len(ods) - len(ids))
            if j >= 0 and ids[j] != 1:
                raise _Mix("expand of a non-unit dimension to S")
            return Verdict(SLICE), {s_out[0]: Layout(axis, m, None)}
        if ty in _FACTORIES:
            out = {}
            for t in s_out:
                axis, m, why = self.s_dim(t)
                if axis is None:
                    raise _Mix(why)
                out[t] = Layout(axis, m, None)
            return Verdict(SLICE), out
        raise _Mix(f"{ty} produces S from no S input")

    def _rule_pointwise(self, ty, op, ins, outs, lays) -> Tuple[Verdict, Dict[str, Layout]]:
        self._no_s_args(op)
        out0 = outs[0]
        axis, m, why = self.s_dim(out0)
        if axis is None:
            raise _Mix(why)
        rank = len(self.dims(out0))
        inner = None
        for t in ins:
            tr = len(self.dims(t))
            if t in lays:
                lay = lays[t]
                if lay.axis + (rank - tr) != axis:
                    raise _Mix(f"{t}'s axis does not align with the output's")
                if lay.inner is not None:
                    if lay.dim != m:
                        raise _Mix(f"{t}'s S extent {lay.dim} differs from the output's {m}")
                    if inner is not None and inner != lay.inner:
                        raise _Mix("the inputs disagree on the axis's inner extent")
                    inner = lay.inner
            else:
                j = axis - (rank - tr)
                if j >= 0:
                    d = self.dims(t)[j]
                    if d != 1:
                        raise _Mix(f"{t} has extent {d} on the axis without carrying S")
        if ty == "native_layer_norm":
            n = len(self._args(op)[1] or [])
            if axis >= rank - n:
                raise _Mix("layer norm over the axis")
        if ty == "custom::rms_norm" and axis >= rank - 1:
            raise _Mix("rms norm over the axis")
        if ty == "_native_batch_norm_legit_no_training" and axis == 1:
            raise _Mix("batch norm channel axis")
        res: Dict[str, Layout] = {}
        for t in outs:
            if not self.carries(t):
                continue
            a, mm, why = self.s_dim(t)
            if a is None:
                raise _Mix(why)
            if mm != m:
                raise _Mix(f"{t}'s S extent differs from {out0}'s")
            res[t] = Layout(a, mm, inner)
        return Verdict(SLICE), res

    def _size_list_ok(self, sizes: Any, b: int, what: str) -> None:
        if not isinstance(sizes, list):
            return
        for j, s in enumerate(sizes):
            hit = mentions(s, self.sym)
            if j == b:
                if not hit and s != -1:
                    raise _Mix(f"{what} names the axis's extent as the literal {s!r}")
            elif hit:
                raise _Mix(f"{what} reads S off the axis")

    def _rule_expand(self, op, src, lay, outs) -> Tuple[Verdict, Dict[str, Layout]]:
        out = outs[0]
        res = self._carry(out, lay)
        if res.axis != lay.axis + len(self.dims(out)) - len(self.dims(src)):
            raise _Mix("expand moved the axis")
        args = self._args(op)
        self._size_list_ok(args[1] if len(args) > 1 else None, res.axis, "expand")
        return Verdict(SLICE), {out: res}

    def _rule_view(self, ty, op, src, lay, outs) -> Tuple[Verdict, Dict[str, Layout]]:
        out = outs[0]
        if lay.inner is None:
            raise _Mix("a view of a broadcast axis")
        ims = self.monos(src)
        oms = self.monos(out)
        if any(x is None for x in ims) or any(x is None for x in oms):
            raise _Mix("a view of a dimension that is not a monomial")
        P = self.p_of(lay)
        if P is None:
            raise _Mix("the axis's outer extent is not exact")
        inner_total, outer_total = lay.inner, P
        for j, d in enumerate(ims):
            if j > lay.axis:
                inner_total = inner_total * d
            elif j < lay.axis:
                outer_total = outer_total * d
        b, mb, why = self.s_dim(out)
        if b is None:
            raise _Mix(why)
        after = ONE
        for d in oms[b + 1:]:
            after = after * d
        r2 = inner_total.div(after)
        if r2 is None:
            raise _Mix("the view splits the axis's inner extent")
        p2 = mb.div(self.S * r2)
        if p2 is None:
            raise _Mix("the view folds the axis unevenly")
        before = p2
        for d in oms[:b]:
            before = before * d
        if before != outer_total:
            raise _Mix("the view moves the axis's outer extent")
        if ty in ("view", "_unsafe_view", "reshape", "unflatten"):
            args = self._args(op)
            sizes = args[1] if ty != "unflatten" else (args[2] if len(args) > 2 else None)
            if ty == "unflatten":
                # Only the unflattened block's sizes are listed: the axis's position inside it.
                d0 = _norm_dim(args[1], len(ims)) if len(args) > 1 else None
                if isinstance(sizes, list) and d0 is not None and d0 <= b < d0 + len(sizes):
                    self._size_list_ok(sizes, b - d0, ty)
                elif mentions(sizes, self.sym):
                    raise _Mix("unflatten reads S off the axis")
            else:
                self._size_list_ok(sizes, b, ty)
        return Verdict(SLICE), {out: Layout(b, mb, r2)}

    def _dim_arg(self, op: Dict[str, Any], rank: int) -> Optional[List[int]]:
        """The dims a dim-taking op works over, or None for all of them."""
        attrs = op.get("attributes") or {}
        args = self._args(op)
        raw = None
        for key in ("dim", "dims"):
            if key in attrs and not isinstance(attrs[key], dict):
                raw = attrs[key]
                break
        if raw is None and len(args) > 1:
            raw = args[1]
        if raw is None:
            return None
        if isinstance(raw, list):
            if not raw:
                return None
            return [d % rank for d in raw if isinstance(d, int)]
        if isinstance(raw, int) and not isinstance(raw, bool):
            return [raw % rank]
        return None

    def _rule_dim(self, ty, op, ins, s_in, lays, outs) -> Tuple[Verdict, Dict[str, Layout]]:
        x = ins[0] if ins else None
        if ty == "index_select":
            idx = ins[1] if len(ins) > 1 else None
            if x in s_in and idx in s_in:
                raise _Mix("index_select with S on both sides")
            if idx in s_in:
                self._no_s_args(op)
                return Verdict(SLICE), {t: self._carry(t, lays[idx]) for t in outs if self.carries(t)}
        if x not in s_in:
            raise _Mix(f"{ty} reads S from a non-primary input")
        lay = lays[x]
        rank = len(self.dims(x))
        args = self._args(op)
        if ty in _DIM_SELECT:
            # aten's positions: split / split_with_sizes / chunk (x, n, dim); slice (x, dim, start,
            # end, step); select / narrow (x, dim, ...); unbind (x, dim).
            pos = 2 if ty in ("split", "split_with_sizes", "chunk") else 1
            attr = (op.get("attributes") or {}).get("dim")
            d = attr if isinstance(attr, int) and not isinstance(attr, bool) else (
                args[pos] if len(args) > pos else 0)
            if not isinstance(d, int) or isinstance(d, bool):
                raise _Mix(f"{ty} along an unknown dimension")
            if lay.axis == d % rank:
                # `x[0:]` along the axis — start 0, no end short of the extent (aten records Python's
                # open end as the int64 maximum), step 1 — is the whole axis: an identity, carried
                if not (ty == "slice" and self._whole_slice(args, op)):
                    raise _Mix(f"{ty} along the axis")
            self._no_s_args(op)
            return Verdict(SLICE), {t: self._carry(t, lay) for t in outs if self.carries(t)}
        dims = self._dim_arg(op, rank)
        over = dims is None or lay.axis in dims
        if ty == "sum" and over:
            if any(self.carries(t) for t in outs):
                raise _Mix("a sum over the axis that keeps it")
            return Verdict(CONTRACT), {}
        if over:
            raise _Mix(f"{ty} over the axis")
        self._no_s_args(op)
        return Verdict(SLICE), {t: self._carry(t, lay) for t in outs if self.carries(t)}

    @staticmethod
    def _whole_slice(args: List[Any], op: Dict[str, Any]) -> bool:
        """Whether aten.slice(x, dim, start, end, step) takes every row: start 0 (or absent), end absent or at
        least the int64 maximum Python's open end is recorded as, step 1 (or absent)."""
        def num(i: int, default: Any) -> Any:
            v = args[i] if len(args) > i else default
            if isinstance(v, dict) and "value" in v:
                v = v["value"]
            return default if v is None else v
        start, end, step = num(2, 0), num(3, None), num(4, 1)
        return start == 0 and step == 1 and (end is None or (isinstance(end, int) and end >= 2 ** 63 - 1))

    def _rule_cat(self, ty, op, s_in, lays, outs) -> Tuple[Verdict, Dict[str, Layout]]:
        args = self._args(op)
        members = [x[1] for x in (args[0] if args and isinstance(args[0], list) else [])]
        if any(t not in s_in for t in members):
            raise _Mix(f"{ty} of a tensor without the axis")
        ref = [lays[t] for t in members]
        rank = len(self.dims(members[0]))
        d = args[1] if len(args) > 1 and isinstance(args[1], int) else 0
        d = d % (rank + (1 if ty == "stack" else 0))
        axes = {lay.axis for lay in ref}
        inners = {lay.inner for lay in ref if lay.inner is not None}
        if len(axes) != 1 or len(inners) > 1:
            raise _Mix(f"{ty} members disagree on the axis")
        if ty == "cat" and d == ref[0].axis:
            raise _Mix("cat along the axis")
        inner = next(iter(inners)) if inners else None
        out = self._carry(outs[0], Layout(ref[0].axis, ref[0].dim, inner))
        return Verdict(SLICE), {outs[0]: out}

    def _rule_matmul(self, ty, op, ins, lays, outs) -> Tuple[Verdict, Dict[str, Layout]]:
        self._no_s_args(op)
        if ty == "linear":
            x = ins[0]
            if x not in lays or any(t in lays for t in ins[1:]):
                raise _Mix("linear with S in its weight")
            if lays[x].axis == len(self.dims(x)) - 1:
                raise _Mix("linear over the axis")
            return Verdict(SLICE), {outs[0]: self._carry(outs[0], lays[x])}
        if ty in ("addmm", "baddbmm"):
            bias, a, b = ins[0], ins[1], ins[2]
        else:
            bias, a, b = None, ins[0], ins[1]
        ra = len(self.dims(a))
        la, lb = lays.get(a), lays.get(b)
        k_a = la is not None and la.axis == ra - 1
        k_b = lb is not None and lb.axis == ra - 2
        if k_a or k_b:
            if not (k_a and k_b) or la.inner is None or lb.inner is None or la.inner != lb.inner \
                    or la.dim != lb.dim:
                raise _Mix("a contraction over the axis with mismatched operands")
            if bias is not None:
                raise _Mix("a biased contraction over the axis adds its bias once per slice")
            if any(self.carries(t) for t in outs):
                raise _Mix("a contraction over the axis that keeps it")
            return Verdict(CONTRACT), {}
        batch = ra == 3
        if la is not None and lb is not None:
            if not (batch and la.axis == 0 and lb.axis == 0 and la.inner == lb.inner):
                raise _Mix("S on both operands' free dimensions")
            lay = la
        else:
            lay = la if la is not None else lb
        if bias is not None and bias in lays:
            raise _Mix("a bias carrying S")
        return Verdict(SLICE), {outs[0]: self._carry(outs[0], lay)}

    def _rule_pad(self, op, src, lay, outs) -> Tuple[Verdict, Dict[str, Layout]]:
        args = self._args(op)
        pads = args[1] if len(args) > 1 and isinstance(args[1], list) else []
        rank = len(self.dims(src))
        j = rank - 1 - lay.axis
        if 2 * j + 1 < len(pads) and (pads[2 * j] != 0 or pads[2 * j + 1] != 0):
            raise _Mix("padding along the axis")
        self._no_s_args(op)
        return Verdict(SLICE), {outs[0]: self._carry(outs[0], lay)}

    def _rule_conv(self, op, ins, s_in, lays, outs) -> Tuple[Verdict, Dict[str, Layout]]:
        x = ins[0]
        if s_in != [x]:
            raise _Mix("a convolution with S in its weight")
        lay = lays[x]
        self._no_s_args(op)
        if lay.axis == 1:
            raise _Mix("a convolution over the channel axis")
        if lay.axis >= 2:
            args = self._args(op)
            w = self.dims(ins[1])
            k = lay.axis - 2
            stride = args[3] if len(args) > 3 and isinstance(args[3], list) else []
            pad = args[4] if len(args) > 4 and isinstance(args[4], list) else []
            transposed = args[6] if len(args) > 6 else False
            one = lambda xs: (xs[k] if k < len(xs) else (xs[0] if len(xs) == 1 else None))
            dil = args[5] if len(args) > 5 and isinstance(args[5], list) else []
            kk, d, p = w[lay.axis], one(dil) or 1, one(pad) or 0
            if transposed or one(stride) not in (1, None) or not isinstance(kk, int):
                raise _Mix("a convolution whose kernel spans the axis with a stride or transposed")
            if kk != 1:
                # a SAME convolution (its padding keeps the extent): output row r reads input rows
                # r - d*(k-1)/2 .. r + d*(k-1)/2 — local, run on a slice widened by that radius
                if (d * (kk - 1)) % 2 or p != d * (kk - 1) // 2:
                    raise _Mix("a convolution whose padding does not keep the axis' extent")
                return Verdict(SLICE, halo=d * (kk - 1) // 2), {outs[0]: self._carry(outs[0], lay)}
            if one(pad) not in (0, None):
                raise _Mix("a pointwise convolution padding the axis")
        return Verdict(SLICE), {outs[0]: self._carry(outs[0], lay)}

    def _rule_im2col(self, op, ins, s_in, lays, outs) -> Tuple[Verdict, Dict[str, Layout]]:
        """Unfold of [N, C, H, W] into [N, C*kh*kw, L], L = out_h * out_w, along a spatial axis read in units
        of the op's stride: output row j reads input rows j*s - p .. j*s - p + d*(k-1), so when one unit of the
        axis is one stride (the input's inner extent along it is s) and the windows tile the extent
        (d*(k-1) + 1 - 2p == s: out = in / s), output unit j reads input unit j widened by p rows on each side —
        local, run on a slice widened by that halo (HAT's overlapping cross-attention: k 24, s 16, p 4).
        Anything else — a stride that is not the unit, windows that do not tile, the axis on channels — mixes."""
        x = ins[0]
        if s_in != [x]:
            raise _Mix("an unfold with S outside its input")
        lay = lays[x]
        self._no_s_args(op, allow_lists=True)
        if lay.axis not in (2, 3) or lay.inner is None:
            raise _Mix("an unfold over a non-spatial axis")
        args = self._args(op)
        k = lay.axis - 2

        def at(i: int) -> Optional[int]:
            xs = args[i] if len(args) > i and isinstance(args[i], list) else []
            v = xs[k] if k < len(xs) else (xs[0] if len(xs) == 1 else None)
            return v if isinstance(v, int) and not isinstance(v, bool) else None
        kk, d, p, st = at(1), at(2), at(3), at(4)
        P = self.p_of(lay)
        if None in (kk, d, p, st) or P is None or not P.is_one() or lay.inner.atoms or lay.inner.coef != st:
            raise _Mix("an unfold whose stride is not one unit of the axis")
        if d * (kk - 1) + 1 - 2 * p != st or d * (kk - 1) - p - (st - 1) != p:
            raise _Mix("an unfold whose windows do not tile the axis' extent")
        out = outs[0]
        b, m, why = self.s_dim(out)
        if b is None:
            raise _Mix(why)
        if b != 2:
            raise _Mix("an unfold whose windows' axis is not its output's last")
        if lay.axis == 2:
            inner = m.div(self.S)
            if inner is None:
                raise _Mix("an unfold whose window count is not the axis times the width")
        else:
            inner = ONE
            if m.div(self.S) is None:
                raise _Mix("an unfold whose window count does not hold the axis")
        return Verdict(SLICE, halo=p), {out: Layout(b, m, inner)}

    def _rule_roll(self, op, src, lay, outs) -> Tuple[Verdict, Dict[str, Layout]]:
        """aten.roll(x, shifts, dims): along any dimension but the axis it is per slice; along the axis output row r
        is input row (r - s) mod extent — local with a halo of |s| rows, read cyclically (Swin's shifted windows roll
        the image by half a window, then back). A roll over the flattened tensor (no dims) mixes."""
        args = self._args(op)
        shifts = args[1] if len(args) > 1 else None
        dims = args[2] if len(args) > 2 else None
        if isinstance(shifts, int):
            shifts = [shifts]
        if isinstance(dims, int):
            dims = [dims]
        if not isinstance(shifts, list) or not isinstance(dims, list) or not dims or len(dims) != len(shifts):
            raise _Mix("a roll over the flattened tensor")
        rank = len(self.dims(src))
        along = [sh for sh, d in zip(shifts, dims) if isinstance(d, int) and d % rank == lay.axis]
        out = {outs[0]: self._carry(outs[0], lay)}
        if not along:
            return Verdict(SLICE), out
        if lay.inner is None or not all(isinstance(sh, int) and not isinstance(sh, bool) for sh in along):
            raise _Mix("a roll along the axis by an amount the graph does not fix")
        return Verdict(SLICE, halo=abs(sum(along)), cyclic=True), out

    def _rule_scale(self, ty, op, src, lay, outs) -> Tuple[Verdict, Dict[str, Layout]]:
        """Nearest upsampling and pixel shuffle along a spatial axis: output row r is input row r // c
        (c the constant factor) — a slice of S keeps mapping to whole output rows, the inner extent
        times c. Any other placement of the axis (channels, a non-constant factor) mixes."""
        self._no_s_args(op, allow_lists=True)
        if lay.axis < 2 or lay.inner is None:
            raise _Mix(f"{ty} over a non-spatial axis")
        axis, m, why = self.s_dim(outs[0])
        if axis is None:
            raise _Mix(why)
        if axis != lay.axis:
            raise _Mix(f"{ty} moves the axis")
        c = m.div(lay.dim)
        if c is None or c.atoms or c.coef < 1:
            raise _Mix(f"{ty} scales the axis by a non-constant factor")
        return Verdict(SLICE), {outs[0]: Layout(axis, m, lay.inner * Mono(c.coef))}

    def _rule_sdpa(self, op, ins, s_in, lays, outs) -> Tuple[Verdict, Dict[str, Layout]]:
        self._no_s_args(op)
        ty = _short(str(op.get("op_type") or ""))
        args = self._args(op)
        kw = (op.get("attributes") or {}).get("kwargs") or {}
        at_causal, at_dropout = _SDPA_FLAGS[ty]
        causal = kw.get("is_causal", args[at_causal] if len(args) > at_causal else False)
        dropout = kw.get("dropout_p", args[at_dropout] if len(args) > at_dropout else 0.0)
        if dropout not in (0, 0.0, None, False):
            raise _Mix("attention with dropout draws random numbers per position")
        q, k, v = ins[0], ins[1], ins[2]
        mask = ins[3] if len(ins) > 3 else None
        rq = len(self.dims(q))
        lq = lays.get(q)
        if lq is None:
            raise _Mix("attention whose keys carry the axis and queries do not")
        if lq.axis == rq - 2:
            if k in lays or v in lays:
                raise _Mix("attention over the axis (its keys carry it)")
            if causal not in (False, None, 0):
                raise _Mix("causal attention masks by absolute query position")
        elif lq.axis < rq - 2:
            for t in (k, v):
                lt = lays.get(t)
                if lt is None or lt.axis != lq.axis or lt.inner != lq.inner:
                    raise _Mix("attention whose keys do not share the batch axis")
        else:
            raise _Mix("attention over the head dimension")
        if mask is not None:
            lm = lays.get(mask)
            rm = len(self.dims(mask))
            aligned = lq.axis + (rm - rq)
            if lm is not None:
                if lm.axis != aligned or (lm.inner is not None and lm.inner != lq.inner):
                    raise _Mix("a mask whose axis does not align with the queries'")
            elif aligned >= 0 and self.dims(mask)[aligned] != 1:
                raise _Mix("a mask with extent on the axis that does not carry S")
        res = {}
        for t in outs:
            if self.carries(t):
                res[t] = self._carry(t, lq)
        return Verdict(SLICE), res


class _Mix(Exception):
    pass


# ---------------------------------------------------------------------------------------------
# Regions: a contiguous stretch run in slices, in passes
# ---------------------------------------------------------------------------------------------

@dataclass
class ChunkPass:
    ops: List[str]
    targets: List[str]
    #: Whether any op in it touches the axis: a pass that does not runs once, whole.
    chunked: bool


@dataclass
class RegionPlan:
    symbol: str
    first: int
    last: int
    first_op: str
    last_op: str
    passes: List[ChunkPass] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    contractions: List[str] = field(default_factory=list)
    layouts: Dict[str, Layout] = field(default_factory=dict)
    #: The halo of the stretch, in units of S: each slice [a, b) runs on [a - margin, b + margin] (clipped at the
    #: axis' true ends, where the ops' own padding is the real one) and keeps [a, b) — the accumulated receptive
    #: field of its local ops (`Verdict.halo` over each op's rows per unit of S), summed exactly and rounded up once:
    #: the field itself, so the kept rows are exact.
    margin: int = 0
    #: The axis' unit (`TokenAxis.unit`): a slice of c is c * unit of the symbol.
    unit: int = 1
    #: The stretch holds a cyclic shift along the axis: every slice reads its margin ROUND the axis' ends.
    cyclic: bool = False


def region_reads(ax: TokenAxis, first: int, last: int) -> Tuple[List[str], Set[str]]:
    produced: Set[str] = set()
    reads: List[str] = []
    for uid in ax.order[first:last + 1]:
        op = ax.ops.get(uid) or {}
        for t in op.get("input_tensor_ids") or []:
            meta = ax.tensors.get(t) or {}
            if t in produced or t in reads or meta.get("is_parameter"):
                continue
            reads.append(t)
        produced.update(op.get("output_tensor_ids") or [])
    return reads, produced


def plan_region(ax: TokenAxis, first: int, last: int,
                protected: FrozenSet[str] = frozenset(),
                carriers: Sequence[str] = ()) -> Tuple[Optional[RegionPlan], str]:
    """The passes that run ops [first, last] in slices of `ax.sym`, or (None, why not).

    `carriers`: the component inputs the region's symbols bind from (`build_segment_graph`
    carries them into every piece); an S-carrying one is sliced with the rest."""
    touched = False
    for i in range(first, last + 1):
        v = ax.verdicts[i]
        if v.kind == MIX:
            return None, f"{ax.order[i]} mixes the axis ({v.reason})"
        if v.kind in (SLICE, CONTRACT):
            touched = True
    if not touched:
        return None, "no op in the stretch carries the axis"
    from fractions import Fraction
    margin = Fraction(0)          # the receptive field in units of S, summed exactly, rounded up once
    for i in range(first, last + 1):
        v = ax.verdicts[i]
        if not v.halo:
            continue
        src = next((t for t in (ax.ops.get(ax.order[i]) or {}).get("input_tensor_ids") or [] if t in ax.layout), None)
        lay = ax.layout.get(src) if src else None
        if lay is None or lay.inner is None or lay.inner.atoms or ax.p_of(lay) is None or not ax.p_of(lay).is_one():
            return None, f"{ax.order[i]} reads rows around its own along an axis folded with another extent"
        margin += Fraction(v.halo, lay.inner.coef)
    cyclic = any(ax.verdicts[i].cyclic for i in range(first, last + 1))
    if cyclic and any(ax.verdicts[i].halo and not ax.verdicts[i].cyclic for i in range(first, last + 1)):
        return None, ("a stretch holding a cyclic shift and a zero-padded local op: its slices would read round the "
                      "axis' ends where the padded op reads zeros")
    if margin and any(ax.verdicts[i].kind == CONTRACT for i in range(first, last + 1)):
        return None, ("a stretch of local ops with a halo holds a contraction: its overlapping slices would "
                      "count the margin rows twice in the partial sums")
    reads, produced = region_reads(ax, first, last)
    outputs = [t for uid in ax.order[first:last + 1]
               for t in ((ax.ops.get(uid) or {}).get("output_tensor_ids") or [])
               if (ax.readers.get(t) or [-1])[-1] > last or t in ax.graph_outputs or t in protected]
    plan = RegionPlan(ax.sym, first, last, ax.order[first], ax.order[last], unit=ax.unit)
    plan.margin = math.ceil(margin)
    plan.cyclic = cyclic
    plan.inputs = reads
    plan.outputs = outputs
    for t in list(reads) + [c for c in carriers if c not in reads]:
        if ax.carries(t):
            lay = ax.layout.get(t) or ax._lay(t)
            if lay is None:
                return None, f"input {t} carries the axis without a layout ({ax.bad.get(t, '')})"
            plan.layouts[t] = lay
    for t in outputs:
        if ax.carries(t):
            lay = ax.layout.get(t)
            if lay is None or lay.inner is None:
                return None, f"output {t} carries the axis as a broadcast, which cannot be reassembled"
            plan.layouts[t] = lay
    # Levels: a contraction's output is complete only after every slice; its readers come later.
    level: Dict[str, int] = {t: 0 for t in reads}
    contraction_out: Set[str] = set()
    producer: Dict[str, int] = {}
    for i in range(first, last + 1):
        op = ax.ops.get(ax.order[i]) or {}
        lv = max([level.get(t, 0) for t in op.get("input_tensor_ids") or []] or [0])
        for t in op.get("output_tensor_ids") or []:
            producer[t] = i
            if ax.verdicts[i].kind == CONTRACT:
                level[t] = lv + 1
                contraction_out.add(t)
            else:
                level[t] = lv
    plan.contractions = sorted(contraction_out, key=lambda t: producer[t])

    def pass_of(t: str) -> int:
        return level[t] - 1 if t in contraction_out else level[t]

    wanted = set(outputs) | contraction_out
    top = max([pass_of(t) for t in wanted] or [0])
    for p in range(top + 1):
        targets = sorted((t for t in wanted if pass_of(t) == p), key=lambda t: producer[t])
        if not targets:
            continue
        fed = set(reads) | {t for t in contraction_out if pass_of(t) < p}
        need: Set[int] = set()
        stack = list(targets)
        seen: Set[str] = set()
        while stack:
            t = stack.pop()
            if t in seen or t in fed or t not in producer:
                continue
            seen.add(t)
            i = producer[t]
            if i in need:
                continue
            need.add(i)
            stack.extend((ax.ops.get(ax.order[i]) or {}).get("input_tensor_ids") or [])
        idx = sorted(need)
        chunked = any(ax.verdicts[i].kind in (SLICE, CONTRACT) for i in idx)
        plan.passes.append(ChunkPass([ax.order[i] for i in idx], targets, chunked))
    return plan, ""


def chunk_extents(total: int, count: int) -> List[Tuple[int, int]]:
    """(start, length) of each slice: `count` slices as even as ceil allows — at most two lengths."""
    if count < 1 or total < 1:
        raise ValueError(f"chunk_extents: {count} slices of {total}")
    c = -(-total // count)
    return [(f, min(c, total - f)) for f in range(0, total, c)]


def slice_reads(total: int, count: int, margin: int, cyclic: bool = False) -> List[Tuple[int, int, int, int]]:
    """(start, length, read start, read length) of each slice: its own rows (`chunk_extents`) and the rows it READS —
    widened by the stretch's halo `margin` on each side, clipped at the axis' true ends; ROUND them when the stretch
    shifts cyclically (`RegionPlan.cyclic`: the read start may be negative or the read run past the end, taken modulo
    the axis by `take_slice_cyclic`), one slice then reading the axis as it is. The executor runs a slice at its read
    length, so the census derives a sliced stretch's keys at the same lengths (one function for both)."""
    out = []
    for start, length in chunk_extents(total, count):
        if cyclic and count > 1:
            out.append((start, length, start - margin, length + 2 * margin))
            continue
        in_start = max(0, start - margin)
        out.append((start, length, in_start, min(total, start + length + margin) - in_start))
    return out


def pass_graph(graph: Dict[str, Any], cp: ChunkPass) -> Dict[str, Any]:
    """The minimal graph a pass's liveness is walked on (sizing only)."""
    ops = graph.get("ops") or {}
    return {"tensors": graph.get("tensors") or {}, "ops": {u: ops[u] for u in cp.ops},
            "execution_order": list(cp.ops), "output_tensor_ids": list(cp.targets),
            "symbolic_context": graph.get("symbolic_context")}


# Runtime helpers: duck-typed over torch.Tensor and NBXTensor (view, reshape, narrow, contiguous,
# new_empty, copy_) — no import of either (R33).

def evaluate_inner(lay: Layout, symbol: Callable[[str], int]) -> int:
    return lay.inner.evaluate(symbol)


def take_slice(x: Any, lay: Layout, total: int, inner: Optional[int], start: int, length: int) -> Any:
    """The slice [start, start+length) of the axis of a WHOLE tensor `x`, contiguous. A broadcast
    (`inner` None) is constant along the axis: its first extent at `length` is the slice —
    contiguous too, as every input an executor is fed (priced in `_region_peak`)."""
    shape = tuple(int(d) for d in x.shape)
    a = lay.axis
    if inner is None:
        n = shape[a] // total * length
        return x.narrow(a, 0, n).contiguous()
    p = shape[a] // (total * inner)
    if p * total * inner != shape[a]:
        raise RuntimeError(f"chunked region: extent {shape[a]} on axis {a} is not P*{total}*{inner}")
    pre, post = shape[:a], shape[a + 1:]
    v = x.reshape(*pre, p, total, inner, *post).narrow(a + 1, start, length).contiguous()
    return v.reshape(*pre, p * length * inner, *post)


def take_slice_cyclic(x: Any, lay: Layout, total: int, inner: Optional[int], start: int, length: int) -> Any:
    """`take_slice` of the rows [start, start + length) taken modulo the axis' extent: a read that runs off either
    end continues from the other (a cyclic stretch's margin, `slice_reads`)."""
    if 0 <= start and start + length <= total:
        return take_slice(x, lay, total, inner, start, length)
    if inner is None:                       # a broadcast is the same along the axis
        return take_slice(x, lay, total, inner, 0, length)
    pieces, at, left = [], start % total, length
    while left > 0:
        n = min(left, total - at)
        pieces.append((take_slice(x, lay, total, inner, at, n), n))
        left -= n
        at = 0
    first, n0 = pieces[0]
    whole = first.new_empty(whole_shape(first.shape, lay, length, n0))
    off = 0
    for part, n in pieces:
        put_slice(whole, part, lay, length, inner, off, n)
        off += n
    return whole


def put_slice(full: Any, part: Any, lay: Layout, total: int, inner: int, start: int, length: int) -> None:
    """Write a slice's output into the whole output (allocated contiguous)."""
    shape = tuple(int(d) for d in full.shape)
    a = lay.axis
    p = shape[a] // (total * inner)
    pre, post = shape[:a], shape[a + 1:]
    dst = full.view(*pre, p, total, inner, *post).narrow(a + 1, start, length)
    dst.copy_(part.reshape(*pre, p, length, inner, *post))


def whole_shape(part_shape: Sequence[int], lay: Layout, total: int, length: int) -> Tuple[int, ...]:
    s = [int(d) for d in part_shape]
    s[lay.axis] = s[lay.axis] // length * total
    return tuple(s)


def own(x: Any, dtype: Any = None) -> Any:
    """A copy the next run of the same executor cannot overwrite (an arena slot is reused), in
    `dtype` when given (`copy_` converts on the store, in both engines)."""
    shape = tuple(int(d) for d in x.shape)
    o = x.new_empty(shape) if dtype is None else x.new_empty(shape, dtype=dtype)
    o.copy_(x)
    return o


def accumulate(acc: Any, part: Any, dtype: Any) -> Any:
    """The running sum of a contraction's partials over the slices: `part` added to `acc` (None at
    the first slice) in `dtype` — the accumulator the executor's DtypeEngine names
    (`accumulation_dtype`), never the pass dtype the partial arrives in."""
    if acc is None:
        return own(part, dtype)
    return acc + (part if part.dtype == dtype else part.to(dtype))


def store(acc: Any, dtype: Any) -> Any:
    """A contraction's sum stored once in `dtype`, the dtype its own output was given."""
    return acc if acc.dtype == dtype else own(acc, dtype)


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def chunk_count(total: int, size: int) -> int:
    return max(1, math.ceil(total / max(1, size)))


def dim_source(info: Any) -> Optional[Tuple[str, int]]:
    """(tensor id, dimension) a symbol binds from, or None (a value-sourced or unknown source)."""
    if not isinstance(info, dict):
        return None
    raw = info.get("source")
    if isinstance(raw, dict):
        tid, dim = raw.get("tensor_id"), raw.get("dim")
        return (str(tid), int(dim)) if tid is not None and isinstance(dim, int) else None
    if isinstance(raw, str) and "::dim_" in raw:
        tid, _, d = raw.rpartition("::dim_")
        return (tid, int(d)) if d.isdigit() else None
    return None


def token_symbols(graph: Dict[str, Any], symbol_map: Dict[str, int]) -> List[str]:
    """The symbols an axis can be sliced along: bound from a component input's DIMENSION (a value
    read off a tensor is not an extent) and at least 2 at this request."""
    syms = ((graph.get("symbolic_context") or {}).get("symbols") or {})
    return sorted(sid for sid, info in syms.items()
                  if dim_source(info) is not None and int(symbol_map.get(sid, 0) or 0) >= 2)


def op_symbols(graph: Dict[str, Any]) -> Dict[str, FrozenSet[str]]:
    """op uid -> the symbols its attributes and its tensors' symbolic shapes mention. Computed once
    per graph: a stretch's carriers are the union over its ops (`region_carriers`)."""
    syms = ((graph.get("symbolic_context") or {}).get("symbols") or {})
    ops = graph.get("ops") or {}
    tensors = graph.get("tensors") or {}
    per_tensor: Dict[str, FrozenSet[str]] = {}

    def walk(obj: Any, used: Set[str]) -> None:
        if isinstance(obj, dict):
            sid = _symbol_id(obj)
            if sid:
                used.add(sid)
            if isinstance(obj.get("symbol_id"), str):
                used.add(obj["symbol_id"])
            for v in obj.values():
                walk(v, used)
        elif isinstance(obj, list):
            for v in obj:
                walk(v, used)
        elif isinstance(obj, str) and obj in syms:
            used.add(obj)

    out: Dict[str, FrozenSet[str]] = {}
    for uid, op in ops.items():
        used: Set[str] = set()
        walk((op or {}).get("attributes"), used)
        for t in list((op or {}).get("input_tensor_ids") or []) + list((op or {}).get("output_tensor_ids") or []):
            if t not in per_tensor:
                got: Set[str] = set()
                walk((tensors.get(t) or {}).get("symbolic_shape"), got)
                per_tensor[t] = frozenset(got)
            used |= per_tensor[t]
        out[uid] = frozenset(used)
    return out


def region_carriers(graph: Dict[str, Any], op_uids: Sequence[str],
                    symbols_of: Optional[Dict[str, FrozenSet[str]]] = None) -> List[str]:
    """The component inputs the symbols of these ops bind from — what `build_segment_graph`
    carries into a piece made of them. `symbols_of`: `op_symbols(graph)`, when the caller asks
    for many stretches of one graph."""
    syms = ((graph.get("symbolic_context") or {}).get("symbols") or {})
    if symbols_of is None:
        symbols_of = op_symbols({"ops": {u: (graph.get("ops") or {}).get(u) for u in op_uids},
                                 "tensors": graph.get("tensors"),
                                 "symbolic_context": graph.get("symbolic_context")})
    used: Set[str] = set()
    for uid in op_uids:
        used |= symbols_of.get(uid, frozenset())
    inputs = set(graph.get("input_tensor_ids") or [])
    out: List[str] = []
    for sid in sorted(used):
        src = dim_source(syms.get(sid))
        if src is not None and src[0] in inputs and src[0] not in out:
            out.append(src[0])
    return out
