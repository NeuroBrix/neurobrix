"""ONE liveness rule (core/runtime/liveness.py) for the placement estimate and every engine that frees.

Four copies existed (Prism's profiler, the Triton sequence, the compiled sequence, the compiled
sequential executor); Prism read `input_tensor_ids` while the engines read the args/kwargs, so
what the simulation priced was not by construction what the runtime freed (stage B P1).
"""
import inspect

from neurobrix.core.runtime import liveness


def _ref(tid):
    return {"type": "tensor", "tensor_id": tid}


def _graph():
    ops = {
        "a": {"attributes": {"args": [_ref("input::x")]}, "output_tensor_ids": ["t1", "stat"]},
        "b": {"attributes": {"args": [{"type": "list", "value": [_ref("t1")]}]}, "output_tensor_ids": ["t2"]},
        "c": {"attributes": {"args": [], "kwargs": {"w": {"type": "tensor_tuple", "tensor_ids": ["t1", "t2"]}}},
              "output_tensor_ids": ["out"]},
    }
    return ["a", "b", "c"], ops


def test_a_tensor_dies_after_its_last_reference():
    order, ops = _graph()
    last = liveness.last_uses(order, ops)
    assert last["t1"] == 2 and last["t2"] == 2 and last["input::x"] == 0


def test_a_never_read_output_dies_at_its_producer():
    order, ops = _graph()
    assert liveness.last_uses(order, ops)["stat"] == 0


def test_a_slot_shared_by_aliases_lives_to_the_latest():
    last = {"t1": 1, "alias": 4, "other": 2}
    assert liveness.slot_last_uses(last, {"t1": 7, "alias": 7, "other": 8}) == {7: 4, 8: 2}


def test_protected_keys_are_never_freed():
    assert liveness.dead_at_op({"w": 0, "t": 0, "u": 3}, {"w"}) == {0: ["t"], 3: ["u"]}


def test_prism_prices_by_the_rule_the_runtime_frees_by(monkeypatch):
    from neurobrix.core.prism import profiler
    order, ops = _graph()
    assert profiler.dag_last_uses({"execution_order": order, "ops": ops}) == {
        tid: order[i] for tid, i in liveness.last_uses(order, ops).items()}
    monkeypatch.setattr(liveness, "last_uses", lambda o, p: {"sentinel": 1})
    assert profiler.dag_last_uses({"execution_order": order, "ops": ops}) == {"sentinel": "b"}


def test_every_engine_frees_by_the_one_rule():
    """A door, not a census: each engine's liveness method calls the shared rule and keeps no loop
    of its own over the outputs (the dead-output rule written a fifth time is the defect)."""
    from neurobrix.core.runtime.graph.compiled_sequence import CompiledSequence
    from neurobrix.core.runtime.graph_executor import GraphExecutor
    from neurobrix.triton.sequence import TritonSequence
    for fn in (CompiledSequence._compute_liveness, TritonSequence._compute_liveness):
        src = inspect.getsource(fn)
        assert "_liveness.last_uses(" in src and '.get("output_tensor_ids"' not in src, fn.__qualname__
    src = inspect.getsource(GraphExecutor)
    assert src.count("_liveness.last_uses(") == 1 and "_collect_arg_tids" not in src
