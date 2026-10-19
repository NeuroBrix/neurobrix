"""Prism prices the invariant state a repeated component keeps through a whole run (state engine step 4).

`core/state/invariant.InvariantState` holds the frontier of the invariant ops from a reusing run's first op, and a copy
of the invariant inputs; the profiler's walk (`ActivationProfiler.estimate_peak_memory`) counts both from op 0 and frees
neither — derived by the engines' own rule (`core/optim/passes/invariant.plan_for_dag`) from the declared inputs the
container reader stamps on the graph.

What this test would do if the code were wrong: the toy's peak falls BEFORE the frontier's producer runs (a large varying
intermediate, reduced and freed first). A walk that counts the frontier only from its producer prices 540 bytes instead
of 588 — red; a walk that forgets the kept copy of the invariant input prices 540 — red.
"""
from __future__ import annotations

from neurobrix.core.optim.passes.invariant import DAG_KEY, plan_for_dag
from neurobrix.core.prism.profiler import ActivationProfiler


def _T(tid):
    return {"type": "tensor", "tensor_id": tid}


def _t(param=False):
    d = {"shape": [4, 3], "dtype": "float32"}
    if param:
        d["is_parameter"] = True
    return d


def _dag(stamped: bool):
    """y = sum(relu(x)) + c @ w, fp32: x and a = relu(x) are [40, 3] (480 bytes), b = sum over rows [3] (12 bytes),
    c, k = c @ w and y are [4, 3] (48 bytes); `c` declared invariant."""
    d = {
        "input_tensor_ids": ["input::x", "input::c"], "output_tensor_ids": ["t::y"],
        "tensors": {"input::x": {**_t(), "shape": [40, 3]}, "input::c": _t(),
                    "param::w": {**_t(True), "shape": [3, 3]},
                    "t::a": {**_t(), "shape": [40, 3]}, "t::b": {**_t(), "shape": [3]}, "t::k": _t(), "t::y": _t()},
        "ops": {"relu": {"op_type": "aten::relu", "attributes": {"args": [_T("input::x")]},
                         "input_tensor_ids": ["input::x"], "output_tensor_ids": ["t::a"]},
                "sum": {"op_type": "aten::sum", "attributes": {"args": [_T("t::a"), 0]},
                        "input_tensor_ids": ["t::a"], "output_tensor_ids": ["t::b"]},
                "mm": {"op_type": "aten::mm", "attributes": {"args": [_T("input::c"), _T("param::w")]},
                       "input_tensor_ids": ["input::c", "param::w"], "output_tensor_ids": ["t::k"]},
                "add": {"op_type": "aten::add", "attributes": {"args": [_T("t::b"), _T("t::k")]},
                        "input_tensor_ids": ["t::b", "t::k"], "output_tensor_ids": ["t::y"]}},
        "execution_order": ["relu", "sum", "mm", "add"],
    }
    if stamped:
        d[DAG_KEY] = ["input::c"]
    return d


def test_the_plan_prism_prices_is_the_engines_rule():
    plan = plan_for_dag(_dag(True))
    assert plan["frontier_slots"] == ["t::k"] and plan["input_slots"] == ["input::c"]
    assert plan_for_dag(_dag(False)) is None


def test_the_frontier_and_the_kept_copy_live_through_the_run():
    plain = ActivationProfiler(_dag(False)).estimate_peak_memory(dtype_bytes=4)
    held = ActivationProfiler(_dag(True)).estimate_peak_memory(dtype_bytes=4)
    assert plain.peak_bytes == 492          # a and b live at the sum, before k exists
    assert held.peak_bytes == 588           # + k from the first op (48) + the kept copy of c (48)
