"""A symbolic dim never binds below the extent the container declares.

Every graph the tracer writes carries a constraint per symbol::

    "s1": {"name": "time", "trace_value": 9,
           "source": "input::args::dim_2",
           "constraints": {"min": 1}}

673 symbols across the 182 cached components, every one with a `min`, and
every one of those mins is 1 — so the rule is simply that no batch, seq_len,
height, width or time may bind to 0 or below.

`SymbolicShapeResolver._bind_symbol` has always CLAIMED to validate this. It read
`symbol_info.get("min", 0)` — a key that is not there, because the tracer
nests it under `constraints` — so the comparison was always against the
default 0, and the validator has never rejected anything in the whole zoo.
A validator that reads an absent key is silent, and silence is
indistinguishable from correctness.

The cost, measured 2026-09-10: Wan2.1-VACE-1.3B bound `time` to 0 from a
single still image. Nothing objected. Sixty ops later the VAE encoder asked
the allocator for -4,860,000 bytes at `aten.convolution::60` and the run was
recorded as an OOM — a memory failure standing in for a shape contract that
was declared, checked, and never enforced.

Run: PYTHONPATH=src python -m pytest tests/unit/runtime/test_symbol_min_constraint.py
"""
from __future__ import annotations

import pytest

# The real VACE vae_encoder symbol table (container built 2026-06-16).
VACE_CONTEXT = {
    "symbols": {
        "s0": {"name": "batch", "trace_value": 1,
               "source": "input::args::dim_0", "constraints": {"min": 1}},
        "s1": {"name": "time", "trace_value": 9,
               "source": "input::args::dim_2", "constraints": {"min": 1}},
    },
    "expressions": {},
}


def _resolver():
    from neurobrix.core.runtime.shape_resolver import SymbolicShapeResolver
    return SymbolicShapeResolver(VACE_CONTEXT)


def test_a_dim_of_zero_is_refused_and_named():
    from neurobrix.core.runtime.shape_resolver import ShapeResolutionError

    r = _resolver()
    with pytest.raises(ShapeResolutionError) as e:
        r._bind_symbol("s1", 0, VACE_CONTEXT["symbols"]["s1"])
    msg = str(e.value)
    # The message has to carry the cause, not just the symbol id: the whole
    # point is that this failure used to surface 60 ops away as a negative
    # byte count with no mention of a symbol at all.
    assert "s1" in msg and "time" in msg
    assert "input::args::dim_2" in msg
    assert "0" in msg and "1" in msg


def test_a_negative_dim_is_refused():
    from neurobrix.core.runtime.shape_resolver import ShapeResolutionError

    r = _resolver()
    with pytest.raises(ShapeResolutionError):
        r._bind_symbol("s0", -2, VACE_CONTEXT["symbols"]["s0"])


def test_the_declared_minimum_itself_binds():
    r = _resolver()
    r._bind_symbol("s1", 1, VACE_CONTEXT["symbols"]["s1"])
    assert r._runtime_values["s1"] == 1


def test_an_ordinary_extent_binds():
    r = _resolver()
    r._bind_symbol("s1", 81, VACE_CONTEXT["symbols"]["s1"])
    assert r._runtime_values["s1"] == 81


def test_a_symbol_without_constraints_is_unconstrained():
    """Back-compat: a graph that declares nothing constrains nothing."""
    from neurobrix.core.runtime.shape_resolver import SymbolicShapeResolver

    r = SymbolicShapeResolver({"symbols": {"s9": {"name": "free"}}, "expressions": {}})
    r._bind_symbol("s9", 0, {"name": "free"})
    assert r._runtime_values["s9"] == 0


def test_the_triton_binder_refuses_the_same_value():
    """R30: the Triton branch binds symbols through its own resolver and must
    reject what the ATen branch rejects, or the two modes disagree on which
    requests are legal."""
    from neurobrix.triton.symbols import SymbolResolver

    r = SymbolResolver(VACE_CONTEXT)
    with pytest.raises(Exception) as e:
        r._bind("s1", 0)
    msg = str(e.value)
    assert "s1" in msg and "time" in msg and "input::args::dim_2" in msg
    r._bind("s1", 1)
    assert r._bindings["s1"] == 1


def test_the_triton_binder_has_exactly_one_write_site():
    """The check lives at the write site, so a second write bypasses it in
    silence. `symbols.py` binds through `_bind` and nowhere else."""
    import ast
    import pathlib

    path = (pathlib.Path(__file__).resolve().parents[3] / "src" / "neurobrix"
            / "triton" / "symbols.py")
    assert path.is_file(), f"triton symbol resolver not found at {path}"
    tree = ast.parse(path.read_text(), filename=str(path))
    writes = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for tgt in node.targets
        if isinstance(tgt, ast.Subscript)
        and isinstance(tgt.value, ast.Attribute) and tgt.value.attr == "_bindings"
    ]
    assert len(writes) == 1, (
        f"{len(writes)} writes to _bindings (lines {writes}) — every one but "
        "the store inside _bind skips the container's declared extent")


def test_the_real_vace_table_refuses_the_extent_that_failed():
    """Integration pin on the container that exposed this: the vae_encoder's
    `time` symbol declares min 1, and the run that failed bound it to 0."""
    import json
    import os

    from neurobrix.core.runtime.shape_resolver import (
        ShapeResolutionError, SymbolicShapeResolver)

    g = os.path.expanduser("~/.neurobrix/cache/Wan2.1-VACE-1.3B-diffusers"
                           "/components/vae_encoder/graph.json")
    if not os.path.exists(g):
        pytest.skip("Wan2.1-VACE-1.3B-diffusers container not extracted here")

    ctx = json.load(open(g)).get("symbolic_context") or {}
    symbols = ctx.get("symbols") or {}
    time_sym = next((sid for sid, m in symbols.items()
                     if (m or {}).get("name") == "time"), None)
    assert time_sym, "the vae_encoder no longer declares a `time` symbol"

    r = SymbolicShapeResolver(ctx)
    with pytest.raises(ShapeResolutionError):
        r._bind_symbol(time_sym, 0, symbols[time_sym])
    r._bind_symbol(time_sym, 9, symbols[time_sym])   # the traced extent
    assert r._runtime_values[time_sym] == 9
