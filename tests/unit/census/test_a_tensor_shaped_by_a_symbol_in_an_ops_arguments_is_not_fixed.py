"""A tensor computed from parameters alone is FIXED (its traced shape, whatever its annotation) —
unless an op on the way carries a symbol in its ARGUMENTS: then the request shapes it.

MiniCPM-o-4_5's resampler repeats its learned query by the image-slice count — `repeat(query,
[1, s0, 1])` then `view([64 * s0, 4096])` — with s0 the batch of slices bound from the tower's
output. Every tensor on that chain is a parameter's child, so the derivation classed the view
FIXED and keyed `addmm (192, 4096, 4096)` at the trace's three slices for a one-slice request
whose run asked (64, 4096, 4096) (the Mac, 2026-10-14 06:18; measured on the CPU with its
profile: the site fed x [1, 1024, 1152], the output resolved [1, 64, 4096], the addmm 192).

What this test does if the code is wrong: the symbol-argument cell fails when the repeated query
stays fixed (red before this change: shape(view) = [192, 4096] at s0 = 1); the parameter cell
fails if a plain parameter chain stops being fixed (Flex's annotated weight must keep its stored
4096); the expression cell fails if an expression node (not a bare symbol) in the arguments is
missed.

Run: PYTHONPATH=src python -m pytest tests/unit/census/test_a_tensor_shaped_by_a_symbol_in_an_ops_arguments_is_not_fixed.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
import derived_census as D  # noqa: E402

S0 = {"type": "symbol", "id": "s0", "trace": 3}
MUL = {"type": "mul", "left": 64, "right": S0, "trace": 192}


def _graph(repeat_arg, view_arg):
    T = {"param::query": {"shape": [64, 4096], "is_parameter": True},
         "input::x": {"shape": [3, 323, 1152], "symbolic_shape": {"dims": [S0, {"type": "symbol", "id": "s1", "trace": 323}, 1152]}},
         "aten.unsqueeze::0::out_0": {"shape": [64, 1, 4096]},
         "aten.repeat::0::out_0": {"shape": [64, 3, 4096], "symbolic_shape": {"dims": [64, S0, 4096], "concrete": [64, 3, 4096]}},
         "aten.view::0::out_0": {"shape": [192, 4096], "symbolic_shape": {"dims": [MUL, 4096], "concrete": [192, 4096]}},
         "aten.t::0::out_0": {"shape": [4096, 4096]}, "param::w": {"shape": [4096, 4096], "is_parameter": True}}
    ops = {"aten.unsqueeze::0": {"op_type": "aten::unsqueeze", "input_tensor_ids": ["param::query"], "output_tensor_ids": ["aten.unsqueeze::0::out_0"],
                                 "attributes": {"args": [{"type": "tensor", "tensor_id": "param::query"}, {"type": "scalar", "value": 1}]}},
           "aten.repeat::0": {"op_type": "aten::repeat", "input_tensor_ids": ["aten.unsqueeze::0::out_0"], "output_tensor_ids": ["aten.repeat::0::out_0"],
                              "attributes": {"args": [{"type": "tensor", "tensor_id": "aten.unsqueeze::0::out_0"}, {"type": "list", "value": repeat_arg}]}},
           "aten.view::0": {"op_type": "aten::view", "input_tensor_ids": ["aten.repeat::0::out_0"], "output_tensor_ids": ["aten.view::0::out_0"],
                            "attributes": {"args": [{"type": "tensor", "tensor_id": "aten.repeat::0::out_0"}, {"type": "list", "value": view_arg}]}},
           "aten.t::0": {"op_type": "aten::t", "input_tensor_ids": ["param::w"], "output_tensor_ids": ["aten.t::0::out_0"],
                         "attributes": {"args": [{"type": "tensor", "tensor_id": "param::w"}]}}}
    return {"tensors": T, "ops": ops, "execution_order": list(ops), "input_tensor_ids": ["input::x"],
            "symbolic_context": {"symbols": {"s0": {"name": "batch", "trace_value": 3, "source": "input::x::dim_0"},
                                             "s1": {"name": "seq_len", "trace_value": 323, "source": "input::x::dim_1"}}}}


def test_a_query_repeated_by_a_symbol_argument_follows_the_request():
    g = _graph([1, S0, 1], [MUL, 4096])
    fixed = D._fixed_tensors(g)
    assert "param::query" in fixed and "aten.unsqueeze::0::out_0" in fixed
    assert "aten.repeat::0::out_0" not in fixed and "aten.view::0::out_0" not in fixed
    shape = D._shape_fn(g, "resampler", {"s0": 1, "s1": 1024}, {}, {})
    assert shape("aten.view::0::out_0") == [64, 4096], shape("aten.view::0::out_0")
    assert shape("aten.repeat::0::out_0") == [64, 1, 4096]


def test_a_parameter_chain_without_a_symbol_argument_stays_fixed():
    g = _graph([1, 3, 1], [192, 4096])          # the same chain with literal arguments: fixed, the traced shape
    fixed = D._fixed_tensors(g)
    assert {"aten.repeat::0::out_0", "aten.view::0::out_0", "aten.t::0::out_0"} <= fixed
    shape = D._shape_fn(g, "resampler", {"s0": 1, "s1": 1024}, {}, {})
    assert shape("aten.view::0::out_0") == [192, 4096] and shape("aten.t::0::out_0") == [4096, 4096]


def test_an_expression_in_the_arguments_counts_as_a_symbol():
    g = _graph([1, 3, 1], [MUL, 4096])           # only the view's argument is symbolic (an expression node)
    fixed = D._fixed_tensors(g)
    assert "aten.repeat::0::out_0" in fixed and "aten.view::0::out_0" not in fixed
