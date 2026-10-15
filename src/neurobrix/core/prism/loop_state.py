"""A request the denoising loop cannot keep is refused by name, before anything runs.

An iterative flow feeds its loop component's output back as its next state (`flow.loop.state_input`): the output
must keep the state's shape, step after step. The graph says whether it does — its state input's dims and its
output's dims are both written in the request's symbols. Allegro's transformer patchifies with a stride-2
convolution and unpatchifies by 2: its output height is ((h - 2) // 2 + 1) * 2, which is h only when the latent
height is even. A 360x640 request (latent 45 rows) ran its first step and died in the scheduler, "45 vs 44"
(the Mac, 2026-10-15 05:30); the vendor's pipeline checks its inputs first.

Read from the graph and the request alone: the state input's dims and the output's evaluated at the request's own
binding (`ActivationProfiler.build_symbol_map`, the binding the plan prices). A mismatch is refused with the nearest
request sizes whose latent extent the loop keeps, found by evaluating the same expressions — no divisor, patch size
or model is written here. Pure arithmetic, torch-free.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

_OPS = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
        "floordiv": lambda a, b: a // b, "mod": lambda a, b: a % b}


def _eval(node, env: Dict[str, int]) -> Optional[int]:
    if isinstance(node, bool):
        return None
    if isinstance(node, int):
        return node
    if not isinstance(node, dict):
        return None
    t = node.get("type")
    if t == "symbol":
        return env.get(node.get("id", node.get("symbol_id")))
    if t in _OPS:
        a, b = _eval(node.get("left"), env), _eval(node.get("right"), env)
        if a is None or b is None or (t in ("floordiv", "mod") and b == 0):
            return None
        return _OPS[t](a, b)
    return None


def _symbol_of(node) -> Optional[str]:
    return node.get("id", node.get("symbol_id")) if isinstance(node, dict) and node.get("type") == "symbol" else None


def state_pairs(graph: Dict[str, Any], state_input: str) -> List[Tuple[int, Any, Any]]:
    """(axis, the state input's dim, the output's dim) for the output that returns the state: the graph output of the
    state input's rank. Empty when the graph names no such input or output."""
    tensors = graph.get("tensors") or {}
    sin = tensors.get(f"input::{state_input}") or {}
    din = ((sin.get("symbolic_shape") or {}).get("dims")) or []
    outs = [tensors.get(o) or {} for o in graph.get("output_tensor_ids") or []]
    cands = [((t.get("symbolic_shape") or {}).get("dims") or []) for t in outs]
    cands = [d for d in cands if len(d) == len(din)]
    if not din or not cands:
        return []
    return [(i, a, b) for i, (a, b) in enumerate(zip(din, cands[0]))]


def mismatches(graph: Dict[str, Any], state_input: str, env: Dict[str, int]) -> List[Tuple[int, str, int, int]]:
    """(axis, symbol, the state's extent, the output's) for every axis whose state input is a symbol and whose output
    the request's binding evaluates to another value."""
    out = []
    for axis, a, b in state_pairs(graph, state_input):
        sym = _symbol_of(a)
        if sym is None or sym not in env:
            continue
        va, vb = _eval(a, env), _eval(b, env)
        if va is not None and vb is not None and va != vb:
            out.append((axis, sym, va, vb))
    return out


def kept_neighbours(graph: Dict[str, Any], state_input: str, env: Dict[str, int], sym: str,
                    span: int = 64) -> Tuple[Optional[int], Optional[int]]:
    """The nearest extents of `sym` below and above its value at which the loop keeps the state (every other symbol
    as bound)."""
    def kept(v):
        e = dict(env, **{sym: v})
        return not any(s == sym for _ax, s, _a, _b in mismatches(graph, state_input, e))
    v0 = env[sym]
    below = next((v for v in range(v0 - 1, max(v0 - span, 0), -1) if kept(v)), None)
    above = next((v for v in range(v0 + 1, v0 + span) if kept(v)), None)
    return below, above


def refuse_a_state_the_loop_cannot_keep(topology: Dict[str, Any], graphs, input_config, model: str = "") -> None:
    """Raise, naming the request sizes the loop keeps, when the loop component's output does not return its state's
    shape at the request's binding. `graphs(comp)` reads a component's graph; the binding is the plan's own."""
    from neurobrix.core.prism.profiler import ActivationProfiler
    flow = (topology or {}).get("flow") or {}
    loop = flow.get("loop") or {}
    state_input = loop.get("state_input")
    if flow.get("type") != "iterative_process" or not state_input:
        return
    scale = getattr(input_config, "vae_scale", None)
    for comp in loop.get("components") or []:
        g = graphs(comp)
        env = ActivationProfiler(g).build_symbol_map(input_config)
        bad = mismatches(g, state_input, env)
        if not bad:
            continue
        syms = (g.get("symbolic_context") or {}).get("symbols") or {}
        lines = []
        for axis, sym, va, vb in bad:
            name = (syms.get(sym) or {}).get("name") or sym
            lo, hi = kept_neighbours(g, state_input, env, sym)
            unit = f" x{scale} = " if scale and name in ("height", "width") else ""
            px = (lambda v: f"{v}{unit}{v * scale}" if unit and v is not None else str(v))
            lines.append(f"{name} ({sym}, axis {axis}): the state is {va} and '{comp}' returns {vb}; the loop keeps "
                         f"{px(lo)} below and {px(hi)} above")
        raise ValueError(f"ZERO FALLBACK: {model or 'this model'}'s denoising loop cannot keep this request's state "
                         f"— its output must return the shape it was given, step after step: " + "; ".join(lines)
                         + ". Ask for a size the loop keeps.")
