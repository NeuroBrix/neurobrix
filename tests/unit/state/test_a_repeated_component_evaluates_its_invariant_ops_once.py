"""A repeated component evaluates the ops of its invariant inputs once per request (state engine step 4).

docs/design/state-engine.md §4.1: the build declares which inputs a repeated component reads unchanged (topology.json
`state`, kind `invariant`); `core/optim/passes/invariant.plan_invariant` derives the ops computed from them alone on the
op list each engine runs; `core/state/invariant.InvariantState` keeps their outputs and serves every later run whose
invariant inputs are BIT-identical.

What these tests would do if the code were wrong:
  * a varying op marked invariant: the second run reads a stale value — the toy graph changes the varying input
    between runs and requires the output to follow it exactly (red on both engines);
  * the reuse test numeric instead of bitwise: -0.0 against a kept +0.0 is reused — red at the sign-of-zero run;
  * a symbol bound by a varying input accepted, an in-place op's slot or an intercepted op planned: the planner rows
    below are red.

Run: CUDA_VISIBLE_DEVICES=<card> PYTHONPATH=src python -m pytest -q tests/unit/state/<this file>
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import pytest

from neurobrix.core.optim.passes.invariant import (
    DAG_KEY, InvariantPlanRefused, invariant_input_names, plan_invariant)


# ---------------------------------------------------------------------------
# The planner, on bare ops
# ---------------------------------------------------------------------------

@dataclass
class _Op:
    op_uid: str
    op_type: str
    all_input_slots: Tuple[int, ...]
    output_slots: Tuple[int, ...]
    kill_slots: Tuple[int, ...] = field(default=())


# slots: 0 weight, 1 invariant input, 2 varying input
def _plan(ops, **kw):
    return plan_invariant(ops, weight_slots=[0], invariant_slots=[1], **kw)


def test_the_ops_of_the_invariant_input_alone_are_planned_and_their_read_outputs_are_the_frontier():
    ops = [_Op("mm", "aten::mm", (1, 0), (3,)), _Op("view", "aten::view", (3,), (4,)),
           _Op("add", "aten::add", (2, 4), (5,))]
    plan = _plan(ops)
    assert plan["op_indices"] == [0, 1] and plan["frontier_slots"] == [4] and plan["input_slots"] == [1]


def test_an_op_reading_a_varying_input_is_never_planned():
    # the add reads the invariant input AND a varying value: planned, its output would be a stale frontier
    ops = [_Op("mm", "aten::mm", (2, 0), (3,)), _Op("add", "aten::add", (1, 3), (4,)), _Op("o", "custom::sink", (2, 4), (5,))]
    assert _plan(ops) is None


def test_an_op_of_weights_alone_is_not_this_state_s():
    ops = [_Op("t", "aten::mul", (0, 0), (3,)), _Op("add", "aten::add", (2, 3), (4,))]
    assert _plan(ops) is None


def test_a_symbol_a_varying_input_binds_keeps_the_op_out():
    ops = [_Op("expand", "aten::expand", (1,), (3,)), _Op("add", "aten::add", (2, 3), (4,))]
    syms = {"expand": {"s_img"}}
    assert _plan(ops, symbols_of=lambda u: syms.get(u, set()), invariant_symbols={"s_txt"}) is None
    syms = {"expand": {"s_txt"}}
    assert _plan(ops, symbols_of=lambda u: syms.get(u, set()), invariant_symbols={"s_txt"})["op_indices"] == [0]


def test_a_slot_an_in_place_op_touches_and_an_intercepted_op_stay_out():
    ops = [_Op("mm", "aten::mm", (1, 0), (3,)), _Op("mul_", "aten::mul_", (3, 2), (6,)),
           _Op("add", "aten::add", (2, 3), (4,))]
    assert _plan(ops) is None
    ops = [_Op("sdpa", "aten::scaled_dot_product_attention", (1, 1, 1), (3,)), _Op("add", "aten::add", (2, 3), (4,))]
    assert _plan(ops, excluded_uids={"sdpa"}) is None
    assert _plan(ops)["op_indices"] == [0]


def test_two_writers_of_one_slot_are_refused_by_name():
    ops = [_Op("a", "aten::mm", (1, 0), (3,)), _Op("b", "aten::mm", (1, 0), (3,))]
    with pytest.raises(InvariantPlanRefused, match="written by two ops"):
        _plan(ops)


def test_the_declared_inputs_are_read_from_the_container_state():
    topo = {"state": {"transformer": [{"kind": "invariant", "inputs": ["encoder_hidden_states"]}],
                      "model.decoder": [{"kind": "kv", "layers": 4}]}}
    assert invariant_input_names(topo, "transformer") == ["encoder_hidden_states"]
    assert invariant_input_names(topo, "model.decoder") == [] and invariant_input_names({}, "x") == []


# ---------------------------------------------------------------------------
# Both engines' sequences, across runs
# ---------------------------------------------------------------------------

def _T(tid):
    return {"type": "tensor", "tensor_id": tid}


def _tensor(shape, param=False):
    d = {"shape": list(shape), "dtype": "float32"}
    if param:
        d.update(is_parameter=True, weight_name="w")
    return d


def _dag():
    """y = x + c @ w — `c` declared invariant, `x` varying."""
    return {
        "component_name": "toy", "torch_dtype": "float32",
        "input_tensor_ids": ["input::x", "input::c"], "output_tensor_ids": ["t::y"],
        "tensors": {"input::x": _tensor((4, 3)), "input::c": _tensor((4, 3)), "param::w": _tensor((3, 3), True),
                    "t::k": _tensor((4, 3)), "t::y": _tensor((4, 3))},
        "ops": {"aten::mm::0": {"op_type": "aten::mm", "attributes": {"args": [_T("input::c"), _T("param::w")],
                                                                      "kwargs": {}},
                                "input_tensor_ids": ["input::c", "param::w"], "output_tensor_ids": ["t::k"]},
                "aten::add::1": {"op_type": "aten::add", "attributes": {"args": [_T("input::x"), _T("t::k")],
                                                                        "kwargs": {}},
                                 "input_tensor_ids": ["input::x", "t::k"], "output_tensor_ids": ["t::y"]}},
        "execution_order": ["aten::mm::0", "aten::add::1"],
        DAG_KEY: ["input::c"],
    }


# (invariant input, expected (evaluations, reuses) after the run)
_RUNS = (("ones", (1, 0)), ("ones", (1, 1)), ("zeros", (2, 1)), ("negzeros", (3, 1)), ("negzeros", (3, 2)))


def _c(np, kind):
    return {"ones": np.ones((4, 3), np.float32), "zeros": np.zeros((4, 3), np.float32),
            "negzeros": -np.zeros((4, 3), np.float32)}[kind]


def test_the_compiled_sequence_reuses_only_bit_identical_invariant_inputs():
    torch = pytest.importorskip("torch")
    import numpy as np
    from neurobrix.core.runtime.graph.compiled_sequence import CompiledSequence
    seq = CompiledSequence(_dag(), torch.device("cpu"), torch.float32)
    seq.compile()
    assert seq._invariant is not None and seq._invariant.op_indices == (0,)
    w = torch.randn(3, 3)
    seq.bind_weights({"param::w": w})
    for kind, counts in _RUNS:
        x, c = torch.randn(4, 3), torch.from_numpy(_c(np, kind))
        seq.bind_inputs({"input::x": x, "input::c": c.clone()})
        seq.run()
        y = seq.gather_outputs(["t::y"])["t::y"]
        assert torch.equal(y, x + c @ w), kind
        assert (seq._invariant.evaluations, seq._invariant.reuses) == counts, kind
    seq.release_invariant_state()
    assert not seq._invariant.holds


def test_the_triton_sequence_reuses_only_bit_identical_invariant_inputs():
    pytest.importorskip("triton")
    import numpy as np
    from nbxtensor.tensor import NBXDtype, NBXTensor
    try:
        from nbxtensor.tensor import DeviceAllocator
        DeviceAllocator.ensure_triton_device(0)
    except Exception:
        pytest.skip("no CUDA card")
    from neurobrix.triton.sequence import TritonSequence
    seq = TritonSequence(_dag(), 0, NBXDtype.float32, stores_fp64=True)
    seq.compile()
    assert seq._invariant is not None and seq._invariant.op_indices == (0,)
    rng = np.random.default_rng(0)
    w = rng.standard_normal((3, 3)).astype(np.float32)
    seq.bind_weights({"w": NBXTensor.from_numpy(w).to("cuda")})
    for kind, counts in _RUNS:
        x, c = rng.standard_normal((4, 3)).astype(np.float32), _c(np, kind)
        seq.bind_inputs({"input::x": NBXTensor.from_numpy(x).to("cuda"),
                         "input::c": NBXTensor.from_numpy(c.copy()).to("cuda")})
        seq.run()
        y = seq.gather_outputs(["t::y"])["t::y"].numpy()
        assert np.allclose(y, x + c @ w, rtol=0, atol=1e-5), kind
        assert (seq._invariant.evaluations, seq._invariant.reuses) == counts, kind
    seq.release_invariant_state()
    assert not seq._invariant.holds


def test_the_diagnostic_gate_plans_nothing(monkeypatch):
    torch = pytest.importorskip("torch")
    from neurobrix.core.runtime.graph.compiled_sequence import CompiledSequence
    monkeypatch.setenv("NBX_DISABLE_STATE_INVARIANT", "1")
    seq = CompiledSequence(_dag(), torch.device("cpu"), torch.float32)
    seq.compile()
    assert seq._invariant is None


def test_the_same_bound_objects_are_reused_without_a_comparison():
    """The fast path: the very same input objects as the kept ones skip the bitwise test (whisper's decoder binds the
    same encoder states at every token — the test cost 1.28 ms per token); new objects are compared, then held.

    What this test would do if the code were wrong: a fast path that compared anyway — red at fast_reuses; a fast path
    keyed on equal bits instead of the object — red at the new-object row (fast_reuses would grow there)."""
    torch = pytest.importorskip("torch")
    from neurobrix.core.runtime.graph.compiled_sequence import CompiledSequence
    seq = CompiledSequence(_dag(), torch.device("cpu"), torch.float32)
    seq.compile()
    w = torch.randn(3, 3)
    seq.bind_weights({"param::w": w})
    c = torch.ones(4, 3)
    calls = []
    orig = seq._invariant._same_bits
    seq._invariant._same_bits = lambda a, b: calls.append(1) or orig(a, b)
    for i, (cin, fast, compared) in enumerate(((c, 0, 0), (c, 1, 0), (c, 2, 0), (c.clone(), 2, 1), (None, 3, 1),
                                               (torch.zeros(4, 3), 3, 2))):
        x = torch.randn(4, 3)
        cur = cin if cin is not None else last
        seq.bind_inputs({"input::x": x, "input::c": cur})
        seq.run()
        last = cur
        assert torch.equal(seq.gather_outputs(["t::y"])["t::y"], x + cur @ w), i
        assert (seq._invariant.fast_reuses, len(calls)) == (fast, compared), (i, seq._invariant.fast_reuses, len(calls))
