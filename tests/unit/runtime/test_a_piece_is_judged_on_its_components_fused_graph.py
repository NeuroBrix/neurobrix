"""A streamed piece takes its component's MoE declaration WITHOUT re-fusing its own slice.

Measured 2026-10-13 on Ming-Lite-Omni-1.5, 16 GB (layer_streaming), both engines: every layer of
its language model keeps THREE modality gates (`aten::topk`, k=6) in the fused graph, bound to the
layer's `custom::moe_fused` op by `gate_group`, 36 ops before it; the plan cut piece 0 after a
layer's gates and before its fused op (native: layer 2, `aten.topk::6/7/8`; triton: layer 9,
`aten.topk::27/28/29`). The piece took the component's declaration (89717071), found its three
gates uncovered IN ITS SLICE, re-ran the declared fusion pass on the slice and refused: "no matcher
recognised an expert block behind it" — the block was in piece 1. The whole graph fuses 28/28
layers (the 32 GB cells, resident, are green). The component's graph, not the slice, is where a
router's coverage is judged; a piece takes the flag.

What this test does if the code is wrong: the seam cell raises the refusal above (red before this
change); the door cell fails if a piece accepts a component whose graph still holds an unfused
router; the whole-executor cell fails if a whole executor stops refusing its own unfused router;
the flag cell fails if a piece holding the fused op no longer takes the component's flag.

Run: PYTHONPATH=src python -m pytest tests/unit/runtime/test_a_piece_is_judged_on_its_components_fused_graph.py
"""
import pytest

from neurobrix.core.runtime.graph_executor import GraphExecutor

GATES = ["aten.topk::0", "aten.topk::1", "aten.topk::2"]


def _gate(uid, k=6):
    return {"op_type": "aten::topk", "input_tensor_ids": [f"{uid}::in"],
            "output_tensor_ids": [f"{uid}::out_0", f"{uid}::out_1"],
            "attributes": {"args": [{"type": "tensor", "tensor_id": f"{uid}::in"},
                                    {"type": "scalar", "value": k}], "kwargs": {}}}


def _fused(gate_group, norm=True):
    return {"op_type": "custom::moe_fused", "input_tensor_ids": [f"{g}::out_1" for g in gate_group],
            "output_tensor_ids": ["moe_fused::block.0::out_0"],
            "attributes": {"norm_topk_prob": norm, "gate_group": list(gate_group)}}


def _dag(ops):
    return {"ops": dict(ops), "execution_order": list(ops), "tensors": {}}


def _component_dag():
    ops = {g: _gate(g) for g in GATES}
    ops["moe_fused::block.0"] = _fused(GATES)
    return _dag(ops)


def _executor(dag, **attrs):
    ex = GraphExecutor.__new__(GraphExecutor)
    ex.mode = "compiled"
    ex.family = "multimodal"
    ex._dag = dag
    ex._persistent_pre_op_callback = None
    ex._post_run_hook = None
    for k, v in attrs.items():
        setattr(ex, k, v)
    return ex


def test_a_piece_cut_between_a_layers_gates_and_its_fused_op_takes_the_declaration():
    base = _executor(_component_dag())
    base.set_moe_config(norm_topk_prob=False)                 # the flow declares on the component
    piece0 = _executor(_dag({g: _gate(g) for g in GATES}), _component_from=base)   # the gates, no block
    piece0._take_the_component_moe_declaration()              # red before: "3 router(s) ... left unfused"
    assert piece0._moe_norm_topk_prob is False
    assert list(piece0._dag["execution_order"]) == GATES      # the slice is not rewritten


def test_a_piece_holding_the_fused_op_takes_the_flag():
    base = _executor(_component_dag())
    base.set_moe_config(norm_topk_prob=False)
    piece1 = _executor(_dag({"moe_fused::block.0": _fused(GATES)}), _component_from=base)
    piece1._take_the_component_moe_declaration()
    assert piece1._dag["ops"]["moe_fused::block.0"]["attributes"]["norm_topk_prob"] is False


def test_a_piece_refuses_a_component_whose_graph_still_holds_an_unfused_router():
    base = _executor(_dag({g: _gate(g) for g in GATES}))      # declared, never fused: no fused op covers the gates
    base._moe_norm_topk_prob = True
    piece = _executor(_dag({GATES[0]: _gate(GATES[0])}), _component_from=base)
    with pytest.raises(RuntimeError, match="component this piece was cut from holds 3 unfused router"):
        piece._take_the_component_moe_declaration()


def test_a_whole_executor_still_refuses_its_own_unfused_router():
    whole = _executor(_dag({GATES[0]: _gate(GATES[0])}))
    with pytest.raises(RuntimeError, match="left unfused"):
        whole.set_moe_config(norm_topk_prob=True)
