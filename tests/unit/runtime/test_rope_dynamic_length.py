"""Unit gate for the triton RoPE dynamic-length promotion (case b).

Regression guard for the deepseek-moe degenerate-decode bug
(2026-08-26): graphs that RECOMPUTE RoPE per forward via
arange(seq_len) -> mul(inv_freq) -> cat -> cos/sin -> slice ->
index(position_ids) had their arange/chain-shapes/slice-end
un-promoted to the STATIC trace value. The table then never grew past
trace_seq_len rows, so the first forward whose absolute position
reached trace_seq_len read out of bounds — garbage rotations,
degenerate logits (observed: same 11 tokens as the compiled oracle,
then "!" to the budget, wall exactly at position 23 = trace_seq_len).

Policy under test:
  - case (b) recomputed chain: arange end / chain view shapes /
    slice end become references to the value-sourced symbol
    `nbx_rope_len` (bound per forward from the LAST element of
    input::position_ids, +1 offset at refs = runtime TOTAL length),
    and the symbol is registered in the dag's symbolic_context;
  - case (a) pre-loaded table (TinyLlama pattern): slice end stays
    pinned to the STATIC full source dim — unchanged behaviour;
  - SymbolResolver binds the val_-1 source and resolves ref+offset
    to last_position + 1.

Runnable:  PYTHONPATH=src python3 -m pytest tests/unit/runtime/test_rope_dynamic_length.py -v
"""
from __future__ import annotations

from neurobrix.triton.promotion import promote_seq_len_scalars
from neurobrix.triton.symbols import SymbolResolver

TRACE = 23


def _sym(tv=TRACE):
    return {"type": "symbol", "symbol_id": "s1", "trace_value": tv}


def _make_case_b_dag():
    """Minimal recomputed-RoPE chain (deepseek-moe pattern)."""
    tensors = {
        "param::w": {"weight_name": "w", "shape": [2304, 96]},
        "param::inv_freq": {"weight_name": "rotary_embed.inv_freq",
                            "shape": [64], "constant": True},
        "ar0": {"shape": [TRACE]},
        "v0": {"shape": [TRACE, 1]},
        "m0": {"shape": [TRACE, 64]},
        "c0": {"shape": [TRACE, 128]},
        "cos0": {"shape": [TRACE, 128]},
        "sl0": {"shape": [TRACE, 128]},
        "input::position_ids": {"shape": [1, TRACE]},
    }
    ops = {
        "aten.arange::0": {
            "op_uid": "aten.arange::0", "op_type": "aten::arange",
            "input_tensor_ids": [], "output_tensor_ids": ["ar0"],
            "attributes": {"args": [_sym()], "kwargs": {}},
        },
        "aten.view::0": {
            "op_uid": "aten.view::0", "op_type": "aten::view",
            "input_tensor_ids": ["ar0"], "output_tensor_ids": ["v0"],
            "attributes": {"args": [
                {"type": "tensor", "tensor_id": "ar0"},
                {"type": "list", "value": [_sym(), {"type": "scalar", "value": 1}]},
            ], "kwargs": {}},
        },
        "aten.mul::0": {
            "op_uid": "aten.mul::0", "op_type": "aten::mul",
            "input_tensor_ids": ["v0", "param::inv_freq"],
            "output_tensor_ids": ["m0"],
            "attributes": {"args": [
                {"type": "tensor", "tensor_id": "v0"},
                {"type": "tensor", "tensor_id": "param::inv_freq"},
            ], "kwargs": {}},
        },
        "aten.cat::0": {
            "op_uid": "aten.cat::0", "op_type": "aten::cat",
            "input_tensor_ids": ["m0"], "output_tensor_ids": ["c0"],
            "attributes": {"args": [
                {"type": "tensor_tuple", "tensor_ids": ["m0", "m0"]},
                {"type": "scalar", "value": -1},
            ], "kwargs": {}},
        },
        "aten.cos::0": {
            "op_uid": "aten.cos::0", "op_type": "aten::cos",
            "input_tensor_ids": ["c0"], "output_tensor_ids": ["cos0"],
            "attributes": {"args": [
                {"type": "tensor", "tensor_id": "c0"}], "kwargs": {}},
        },
        "aten.slice::0": {
            "op_uid": "aten.slice::0", "op_type": "aten::slice",
            "input_tensor_ids": ["cos0"], "output_tensor_ids": ["sl0"],
            "attributes": {"args": [
                {"type": "tensor", "tensor_id": "cos0"},
                {"type": "scalar", "value": 0},
                {"type": "scalar", "value": 0},
                _sym(),
            ], "kwargs": {}},
        },
        "aten.index::0": {
            "op_uid": "aten.index::0", "op_type": "aten::index",
            "input_tensor_ids": ["sl0", "input::position_ids"],
            "output_tensor_ids": ["ix0"],
            "attributes": {"args": [
                {"type": "tensor", "tensor_id": "sl0"},
                {"type": "tensor_tuple",
                 "tensor_ids": ["input::position_ids"]},
            ], "kwargs": {}},
        },
    }
    dag = {
        "symbolic_context": {
            "symbols": {"s1": {"name": "seq_len", "trace_value": TRACE}},
        },
        "tensors": tensors,
        "ops": ops,
    }
    return dag, tensors, ops


def _make_case_a_dag():
    """Pre-loaded max-pos table (TinyLlama pattern)."""
    tensors = {
        "param::w": {"weight_name": "w", "shape": [2304, 96]},
        "buffer::cs": {"weight_name": "cos_cached", "shape": [4096, 128]},
        "slA": {"shape": [TRACE, 128]},
        "input::position_ids": {"shape": [1, TRACE]},
    }
    ops = {
        "aten.slice::0": {
            "op_uid": "aten.slice::0", "op_type": "aten::slice",
            "input_tensor_ids": ["buffer::cs"], "output_tensor_ids": ["slA"],
            "attributes": {"args": [
                {"type": "tensor", "tensor_id": "buffer::cs"},
                {"type": "scalar", "value": 0},
                {"type": "scalar", "value": 0},
                _sym(),
            ], "kwargs": {}},
        },
        "aten.index::0": {
            "op_uid": "aten.index::0", "op_type": "aten::index",
            "input_tensor_ids": ["slA", "input::position_ids"],
            "output_tensor_ids": ["ixA"],
            "attributes": {"args": [
                {"type": "tensor", "tensor_id": "slA"},
                {"type": "tensor_tuple",
                 "tensor_ids": ["input::position_ids"]},
            ], "kwargs": {}},
        },
    }
    dag = {
        "symbolic_context": {
            "symbols": {"s1": {"name": "seq_len", "trace_value": TRACE}},
        },
        "tensors": tensors,
        "ops": ops,
    }
    return dag, tensors, ops


def _is_rope_ref(arg, offset=1):
    return (isinstance(arg, dict) and arg.get("type") == "symbol"
            and (arg.get("symbol_id") or arg.get("id")) == "nbx_rope_len"
            and arg.get("offset", 0) == offset)


def test_case_b_chain_goes_dynamic():
    dag, tensors, ops = _make_case_b_dag()
    promote_seq_len_scalars(dag, tensors, ops)

    ar = ops["aten.arange::0"]["attributes"]["args"][0]
    assert _is_rope_ref(ar), f"rope arange end must be dynamic: {ar}"

    sl = ops["aten.slice::0"]["attributes"]["args"][3]
    assert _is_rope_ref(sl), f"case-(b) slice end must be dynamic: {sl}"

    shape = ops["aten.view::0"]["attributes"]["args"][1]
    items = shape.get("value") if isinstance(shape, dict) else shape
    assert _is_rope_ref(items[0]), (
        f"chain view seq dim must be dynamic: {items[0]}")

    syms = dag["symbolic_context"]["symbols"]
    assert "nbx_rope_len" in syms
    assert syms["nbx_rope_len"]["source"] == "input::position_ids::val_-1"
    assert syms["nbx_rope_len"]["trace_value"] == TRACE - 1


def test_case_b_idempotent():
    dag, tensors, ops = _make_case_b_dag()
    promote_seq_len_scalars(dag, tensors, ops)
    promote_seq_len_scalars(dag, tensors, ops)
    ar = ops["aten.arange::0"]["attributes"]["args"][0]
    assert _is_rope_ref(ar), f"second pass must be a no-op: {ar}"


def test_case_a_table_stays_static_full():
    dag, tensors, ops = _make_case_a_dag()
    promote_seq_len_scalars(dag, tensors, ops)
    sl = ops["aten.slice::0"]["attributes"]["args"][3]
    assert sl == {"type": "scalar", "value": 4096}, (
        f"case-(a) slice end must stay the static full table dim: {sl}")
    assert "nbx_rope_len" not in dag["symbolic_context"]["symbols"]


def test_resolver_binds_last_position_plus_one():
    ctx = {"symbols": {"nbx_rope_len": {
        "name": "nbx_rope_len",
        "source": "input::position_ids::val_-1",
        "trace_value": TRACE - 1}}}
    r = SymbolResolver(ctx)
    # prefill: positions 0..12 -> binding 12, ref resolves to 13
    r.bind_from_inputs({"input::position_ids": [0, 1, 2, 3, 4, 5, 6, 7,
                                               8, 9, 10, 11, 12]},
                       [], {})
    assert r.get("nbx_rope_len") == 12
    ref = {"type": "symbol", "id": "nbx_rope_len", "offset": 1,
           "trace": TRACE - 1}
    assert r.resolve(ref) == 13
    # decode at absolute position 40 -> total 41
    r.bind_from_inputs({"input::position_ids": [40]}, [], {})
    assert r.resolve(ref) == 41


def test_unbound_ref_falls_back_to_trace_length():
    r = SymbolResolver({"symbols": {}})
    ref = {"type": "symbol", "id": "nbx_rope_len", "offset": 1,
           "trace": TRACE - 1}
    assert r.resolve(ref) == TRACE


def test_compiled_binder_parses_negative_val_index():
    """Mirror contract: the compiled binder (SymbolicShapeResolver)
    must parse the same value-source vocabulary as the triton binder —
    including the negative index form "val_-1" introduced by
    nbx_rope_len. Before the regex widening (val_(-?\\d+)) it silently
    fell through, leaving the symbol unbound on one engine only."""
    import numpy as np
    from neurobrix.core.runtime.shape_resolver import SymbolicShapeResolver

    ctx = {"symbols": {"nbx_rope_len": {
        "name": "nbx_rope_len",
        "source": "input::position_ids::val_-1",
        "trace_value": TRACE - 1}}}
    r = SymbolicShapeResolver(ctx)
    r.bind_from_inputs({"position_ids": np.array([[0, 1, 2, 3, 40]])}, {})
    assert r.get_bound_symbols().get("nbx_rope_len") == 40


def test_binders_reject_out_of_range_negative_index():
    """Both binders bound-check negative indices (-n <= idx < n)."""
    import numpy as np
    import pytest
    from neurobrix.core.runtime.shape_resolver import (
        ShapeResolutionError, SymbolicShapeResolver)

    ctx = {"symbols": {"x": {"name": "x",
                             "source": "input::t::val_-5",
                             "trace_value": 0}}}
    with pytest.raises(ShapeResolutionError):
        SymbolicShapeResolver(ctx).bind_from_inputs(
            {"t": np.array([1, 2, 3])}, {})

    r = SymbolResolver(ctx)
    with pytest.raises(RuntimeError):
        r.bind_from_inputs({"input::t": [1, 2, 3]}, [], {})
