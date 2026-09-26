"""The MoE fusion finds expert weights by the tokens the NeuroTax parser emits.

It spelled `expert.N.gate.weight` itself. NeuroTax 5.0 names the SwiGLU gate `ffn_gate` (the
vendors' `gate` is the router): a reader with its own spelling would match no expert and the
fusion would switch itself off. The expert keys are built here FROM the parser, so this test
follows the parser wherever it goes; on the old reader it fails (no gate found, the missing
expert synthesized under `gate`).
"""
from __future__ import annotations

from neurobrix.core.runtime.graph import moe_fusion as M
from neurobrix.nbx.neurotax import normalize_tensor_name


def _graph(present_experts, num_experts):
    ops, tensors, producer, order = {}, {}, {}, []
    for e in present_experts:
        for role, vendor in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
            key = normalize_tensor_name(f"model.layers.1.mlp.experts.{e}.{vendor}.weight")
            w = f"param::{key}"
            tensors[w] = {"shape": [8, 4]}
            t_uid, mm_uid = f"aten.t::{e}{role}", f"aten.mm::{e}{role}"
            ops[t_uid] = {"op_type": "aten::t", "attributes": {"args": [{"type": "tensor", "tensor_id": w}]},
                          "input_tensor_ids": [w], "output_tensor_ids": [f"{t_uid}::out_0"]}
            producer[f"{t_uid}::out_0"] = t_uid
            ops[mm_uid] = {"op_type": "aten::mm", "attributes": {"args": [
                {"type": "tensor", "tensor_id": "x"}, {"type": "tensor", "tensor_id": f"{t_uid}::out_0"}]}}
            order += [t_uid, mm_uid]
    ids, found, _ = M._trace_expert_blocks(ops, order, tensors, {}, producer, set(order), "topk", num_experts)
    return ids, found, tensors


def test_present_and_missing_experts_carry_the_parsers_tokens():
    ids, found, tensors = _graph([0, 2], 3)
    assert found == 2
    want = {role: normalize_tensor_name(f"model.layers.1.mlp.experts.1.{v}.weight")
            for role, v in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj"))}
    for role in ("gate", "up", "down"):
        assert len(ids[role]) == 3 and all(ids[role]), ids
        assert ids[role][0] == "param::" + normalize_tensor_name(
            f"model.layers.1.mlp.experts.0.{dict(gate='gate_proj', up='up_proj', down='down_proj')[role]}.weight")
        assert ids[role][1] == "param::" + want[role]      # expert 1 never traced: synthesized
        assert ids[role][1] in tensors
