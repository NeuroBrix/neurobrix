"""A mixture-of-experts layer whose experts live in ONE stacked parameter per
projection is fused like one whose experts are named parameters.

granite-3.1-1b-a400m-instruct (2026-09-20) keeps `input_linear` as [E, 2I, H]
(each expert's gate rows then its up rows) and `output_linear` as [E, H, I], and
reads expert e through `select(param, 0, e) → t → mm`. Its routing weights are a
softmax OVER the selected k (after topk) and the gathered tokens are split per
expert by data-dependent sizes. The fusion pass admitted none of that: the walk
from topk stopped at 12 ops, found no expert, fused nothing, and the
trace-frozen split ran and failed at any prompt length other than the trace's.

Measured on granite's own container: 0 fused layers before, 24 after — every
layer, 32 experts, top-k 8, gate and up handed to the fused op as [I, H]
row-slices of the expert's select. The cell skips where the container is
absent; it asks the pass, not the kernels.
"""
from __future__ import annotations

import json

import pytest

from neurobrix.core.paths import cache_dir

MODEL = "granite-3.1-1b-a400m-instruct"


def _dag():
    p = cache_dir() / MODEL / "components" / "model" / "graph.json"
    if not p.exists():
        pytest.skip(f"{MODEL} is not in this machine's cache")
    return json.loads(p.read_text())


def test_every_layer_of_a_stacked_expert_model_is_fused():
    from neurobrix.core.runtime.graph import moe_fusion as MF
    dag = _dag()
    layers = sum(1 for u in dag["ops"] if u == "aten.topk::0" or u.startswith("aten.topk::"))
    out = MF.detect_and_fuse_moe(dag, "llm", norm_topk_prob=True, declared=True)
    fused = [u for u in out["execution_order"] if u.startswith("moe_fused::")]
    assert len(fused) == layers and layers > 0, (
        f"{len(fused)} of {layers} MoE layers fused — a stacked-expert layer "
        f"must fuse like a named-expert one")
    f = out["ops"][fused[0]]["attributes"]
    assert f["num_experts"] == 32 and f["top_k"] == 8
    # gate and up are ROW SLICES of the expert's input slab, down the output slab
    gate_uid = f["expert_gate_weight_ids"][0].rsplit("::out_0", 1)[0]
    up_uid = f["expert_up_weight_ids"][0].rsplit("::out_0", 1)[0]
    g, u = out["ops"][gate_uid], out["ops"][up_uid]
    assert g["op_type"] == "aten::slice" and u["op_type"] == "aten::slice"
    g_lo, g_hi = g["attributes"]["args"][2]["value"], g["attributes"]["args"][3]["value"]
    u_lo, u_hi = u["attributes"]["args"][2]["value"], u["attributes"]["args"][3]["value"]
    assert (g_lo, g_hi) == (0, 512) and (u_lo, u_hi) == (512, 1024), (g_lo, g_hi, u_lo, u_hi)
    assert g["output_shapes"] == [[512, 1024]]
    # both slices come AFTER the select they view, before the fused op
    order = out["execution_order"]
    sel_uid = g["input_tensor_ids"][0].rsplit("::out_0", 1)[0]
    assert out["ops"][sel_uid]["op_type"] == "aten::select"
    assert order.index(sel_uid) < order.index(gate_uid) < order.index(fused[0])
    assert order.index(sel_uid) < order.index(up_uid) < order.index(fused[0])
    assert f["expert_down_weight_ids"][0].startswith("aten.select::")
