"""ZERO FALLBACK for symbols (the owner, 2026-09-21): a symbol the runtime could not bind
must REFUSE by name, never answer its trace value. Two defects of one day hid behind that
fallback — Wan T2V's plan bound the VAE's spatial symbols to nothing and fell to the trace
extent (1.74 GiB for a 35 GiB decode), and Qwen3-Omni's ATen sequential path evaluated
`s1` at its trace value 23 while the context was 213 (`view [-1, 1, 23]` on 639 elements).
A refusal naming the symbol would have exposed both in a second.

Each cell below binds NOTHING for the symbol and asks for it. What the test would do if the
code were wrong: return 23 (the trace value) — which is what every one of these sites did.
The zero-bound cell: a symbol legitimately bound to 0 (a cache length at its first step) is
BOUND, and must answer 0, never the trace value.
"""
from __future__ import annotations

import pytest

SYM = {"type": "symbol", "id": "s1", "trace": 23, "trace_value": 23}
CTX = {"symbols": {"s1": {"name": "seq_len", "trace_value": 23, "source": "input::inputs_embeds::dim_1"}}}


# ---------------------------------------------------------------- the ATen shape resolver
def test_the_aten_shape_resolver_refuses_an_unbound_symbol():
    from neurobrix.core.runtime.shape_resolver import SymbolicShapeResolver, ShapeResolutionError
    r = SymbolicShapeResolver(CTX)
    r.bind_from_inputs({}, {})                    # the flow provided no inputs_embeds
    with pytest.raises(ShapeResolutionError, match="s1"):
        r.resolve(SYM)


def test_the_aten_shape_resolver_answers_a_symbol_bound_to_zero():
    from neurobrix.core.runtime.shape_resolver import SymbolicShapeResolver
    r = SymbolicShapeResolver(CTX)
    r._bind_symbol("s1", 0, CTX["symbols"]["s1"])
    assert r.resolve(SYM) == 0


# ---------------------------------------------------------------- the Triton symbol resolver
def test_the_triton_symbol_resolver_refuses_an_unbound_symbol():
    from neurobrix.triton.symbols import SymbolResolver, UnboundSymbolError
    r = SymbolResolver(CTX)
    with pytest.raises(UnboundSymbolError, match="s1"):
        r.resolve(SYM)


def test_the_triton_symbol_resolver_answers_a_symbol_bound_to_zero():
    from neurobrix.triton.symbols import SymbolResolver
    r = SymbolResolver(CTX)
    r._bind("s1", 0)
    assert r.resolve(SYM) == 0
    assert r.is_bound("s1") and not r.is_bound("s9")
