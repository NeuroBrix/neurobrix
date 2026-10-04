"""The symbolic expression vocabulary — one grammar, one evaluator, for every resolver.

A container's graph.json carries symbolic values as JSON trees: in `symbolic_shape.dims` (a
tensor's extents), in the size lists of view/reshape/expand/factory ops, and at the scalar slots
of ops whose value follows the request (a `linspace` bound, an `arange` end). Forge emits them
(`SymInt.to_json` for integers, `tracer/symbolic/extent_provenance.py` for reals); every runtime
reader evaluates them HERE — the ATen shape resolver (`shape_resolver.py`), the Triton symbol
resolver (`triton/symbols.py`), both compiled sequences' expression arguments, both sequential
argument paths, Prism's profiler and the census derivation through the Triton resolver. R30: one
vocabulary, one evaluation, both engines. R33: pure Python, no torch.

Nodes (census of the 166 cached graphs, 2026-10-04: symbol, add, sub, mul, floordiv, mod, neg):
  int | float                                    a literal
  {"type": "symbol", "id"|"symbol_id": sN, "offset"?: k}
                                                 a bound symbol (+ k)
  {"type": add|sub|mul|floordiv|mod|truediv, "left": n, "right": n}
                                                 Python's binary operators
  {"type": neg|sqrt, "operand": n}               unary minus; math.sqrt
  {"type": "product", "factors": [n, ...]}       the Triton promotion's product
  {"type": "const"|"scalar", "value": v} | {"value": v}
                                                 a literal wrapped in a dict
  a bare string                                  a symbol id

`truediv` and `sqrt` are REAL: they give a Python float (Forge records the eager arithmetic the
vendor ran — MochiRoPE's `+-height * (area / (height * width)) ** 0.5 / 2`). The evaluation follows
Python's semantics exactly, so a recorded expression evaluates at the trace grid to the value the
trace saw.

Every evaluation names its SLOT. A SHAPE slot (a tensor extent, a size-list element) must give an
int; a real reaching it refuses by name (`RealInShapeSlot`) — an extent can never silently become
a float. A SCALAR slot (an op's top-level scalar argument) takes the int or float the tree gives.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

SHAPE = "shape"
SCALAR = "scalar"

INTEGER_BINARY = ("add", "sub", "mul", "floordiv", "mod")
REAL_BINARY = ("truediv",)
UNARY = ("neg", "sqrt")
REAL_TYPES = frozenset({"truediv", "sqrt"})
# The node types that make a dict an EXPRESSION — a value the runtime computes from the symbols.
# `symbol` is a leaf and every caller already routes it on its own; it is listed separately.
EXPRESSION_TYPES = frozenset(INTEGER_BINARY + REAL_BINARY + UNARY + ("product",))
LITERAL_TYPES = frozenset({"const", "scalar"})

_BINARY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "floordiv": lambda a, b: a // b,
    "mod": lambda a, b: a % b,
    "truediv": lambda a, b: a / b,
}


class ExpressionError(ValueError):
    """A symbolic expression that cannot be evaluated — named, never answered with a guess."""


class RealInShapeSlot(ExpressionError):
    """A real value (true division, square root, a float literal) reached a shape slot."""


class UnknownExpression(ExpressionError):
    """A node type outside the vocabulary."""


def is_expression(node: Any) -> bool:
    """True for a dict the runtime must COMPUTE (an operator node of the vocabulary)."""
    return isinstance(node, dict) and node.get("type") in EXPRESSION_TYPES


def is_symbolic(node: Any) -> bool:
    """True for a symbol leaf or an expression node."""
    return isinstance(node, dict) and (node.get("type") == "symbol" or node.get("type") in EXPRESSION_TYPES)


def real_node(node: Any) -> Optional[str]:
    """The type of the first REAL node in a tree (`truediv`, `sqrt`, or `float` for a float
    literal leaf), None for an integer tree."""
    if isinstance(node, float):
        return "float"
    if isinstance(node, dict):
        t = node.get("type")
        if t in REAL_TYPES:
            return t
        for key in ("left", "right", "operand"):
            if key in node:
                found = real_node(node[key])
                if found:
                    return found
        for f in node.get("factors", ()) or ():
            found = real_node(f)
            if found:
                return found
    return None


def _show(node: Any) -> str:
    text = repr(node)
    return text if len(text) <= 240 else text[:237] + "..."


def _eval(node: Any, symbol: Callable[[str, Any], Any]):
    if isinstance(node, bool):
        raise ExpressionError(f"a boolean is not a symbolic value: {node!r}")
    if isinstance(node, (int, float)):
        return node
    if isinstance(node, str):
        return symbol(node, None)
    if not isinstance(node, dict):
        raise ExpressionError(f"cannot evaluate a {type(node).__name__} as a symbolic value: {_show(node)}")
    t = node.get("type", "")
    if t == "symbol":
        sid = node.get("id") or node.get("symbol_id")
        if not sid:
            raise ExpressionError(f"a symbol node without an id: {_show(node)}")
        return symbol(sid, node) + node.get("offset", 0)
    if t in _BINARY:
        if "left" not in node or "right" not in node:
            raise ExpressionError(f"a {t!r} node needs left and right: {_show(node)}")
        left = _eval(node["left"], symbol)
        right = _eval(node["right"], symbol)
        try:
            return _BINARY[t](left, right)
        except ZeroDivisionError:
            raise ZeroDivisionError(
                f"symbolic {t} by zero ({left!r} {t} {right!r}) in {_show(node)}") from None
    if t in UNARY:
        operand = node.get("operand", node.get("left"))
        if operand is None:
            raise ExpressionError(f"a {t!r} node without an operand: {_show(node)}")
        value = _eval(operand, symbol)
        if t == "neg":
            return -value
        if value < 0:
            raise ExpressionError(f"sqrt of a negative value {value!r} in {_show(node)}")
        return math.sqrt(value)
    if t == "product":
        factors = node.get("factors") or []
        if not factors:
            raise ExpressionError(f"a product expression with no factors: {_show(node)}")
        result = 1
        for f in factors:
            result *= _eval(f, symbol)
        return result
    if t in LITERAL_TYPES or (t == "" and "value" in node):
        value = node.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ExpressionError(f"a literal node that holds no number: {_show(node)}")
        return value
    raise UnknownExpression(
        f"ZERO FALLBACK: expression node of type {t!r} is outside the symbolic vocabulary "
        f"(symbol, {', '.join(sorted(EXPRESSION_TYPES))}): {_show(node)}")


def evaluate(node: Any, symbol: Callable[[str, Any], Any], slot: str):
    """Evaluate `node` with `symbol(id, node_or_None) -> int` answering (or refusing, with the
    caller's own error) each symbol, for a `SHAPE` or `SCALAR` slot."""
    if slot not in (SHAPE, SCALAR):
        raise ValueError(f"unknown slot {slot!r}; a slot is {SHAPE!r} or {SCALAR!r}")
    value = _eval(node, symbol)
    if slot == SHAPE and (isinstance(value, bool) or not isinstance(value, int)):
        raise RealInShapeSlot(
            f"ZERO FALLBACK: a shape slot evaluated to {value!r} ({type(value).__name__}) — the "
            f"expression carries a real node ({real_node(node) or type(value).__name__}); an extent "
            f"is an integer, never a float: {_show(node)}")
    return value
