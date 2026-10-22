"""A weight's `aten::t` is eliminated at bind time (the weight pre-transposed, its readers rewired to it) ONLY when the
graph does not return it — the rule Prism prices by (`profiler.weight_transposes_read_in_place`). A streamed piece
returns every value its seam carries: HAT-S at 2048 px on 16 GB was cut between a linear's weight transpose and its
matmul, the transpose was eliminated in its piece, and the next piece found no `aten.t::0::out_0` (2026-10-22). Both
engines' sequences, one rule.

Injection (seen red, then restored green, 2026-10-22): the `returned` guard removed from either engine -> its case red."""
from __future__ import annotations


import pytest


def _dag(returned):
    ops = {
        "aten.t::0": {"op_uid": "aten.t::0", "op_type": "aten::t",
                      "input_tensor_ids": ["param::w"], "output_tensor_ids": ["aten.t::0::out_0"],
                      "attributes": {"args": [{"type": "tensor", "tensor_id": "param::w"}]}},
        "aten.mm::0": {"op_uid": "aten.mm::0", "op_type": "aten::mm",
                       "input_tensor_ids": ["input::x", "aten.t::0::out_0"], "output_tensor_ids": ["aten.mm::0::out_0"],
                       "attributes": {"args": [{"type": "tensor", "tensor_id": "input::x"},
                                               {"type": "tensor", "tensor_id": "aten.t::0::out_0"}]}},
    }
    tensors = {"param::w": {"shape": [4, 8]}, "input::x": {"shape": [2, 8]},
               "aten.t::0::out_0": {"shape": [8, 4]}, "aten.mm::0::out_0": {"shape": [2, 4]}}
    outs = ["aten.mm::0::out_0"] + (["aten.t::0::out_0"] if returned else [])
    return {"ops": ops, "tensors": tensors, "execution_order": ["aten.t::0", "aten.mm::0"], "output_tensor_ids": outs}


def _engines():
    from neurobrix.triton.sequence import TritonSequence
    from neurobrix.core.runtime.graph.compiled_sequence import CompiledSequence
    return [TritonSequence, CompiledSequence]


@pytest.mark.parametrize("engine", _engines(), ids=lambda c: c.__name__)
@pytest.mark.parametrize("returned", [False, True])
def test_a_weight_transpose_is_eliminated_only_when_not_returned(engine, returned):
    dag = _dag(returned)
    me = engine.__new__(engine)        # the sequence itself, its build skipped: only the bind-time pass runs
    me.dag = dag
    me._eliminate_weight_transpose_ops(dag["tensors"], dag["ops"], dag["execution_order"],
                                       returned=set(dag["output_tensor_ids"]))
    eliminated = "param::w" in (getattr(me, "_pretranspose_weights", None) or set())
    assert eliminated is (not returned), (engine.__name__, returned)
