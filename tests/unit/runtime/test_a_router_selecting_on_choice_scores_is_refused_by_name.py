"""A MoE router that selects its experts on choice scores is refused by name, and its group top-k's are not routers.

The sigmoid grouped router (score = sigmoid(router mm), a choice bias of the expert count added for the selection
only, a group mask that zeroes the experts of the groups not kept, the final top-k on those masked scores, the
weights GATHERED from the unbiased scores, renormalised and scaled) selects on scores it never weights with. Both
matchers of the recogniser bind the top-k input as the gate scores, and the fused dispatch re-selects and
renormalises them: the biased, masked scores would become the weights, without the scale, and no error. Until a
routing op that reads this cluster is bound, the recogniser refuses it, by structure, with a reason that says what
is missing; it never hands it to a matcher.

The router is the top-k that reaches the dispatch. The group cluster's own top-k's (top-2 inside each group, the
group top-k) are part of that router: they are not listed as routers left unfused, at n_group 1 (the group top-k
keeps one group, k = 1) as at n_group 8 (it keeps 4, k = 4).

What these tests would do if the code were wrong:
  * before this refusal, the per-expert matcher was handed the router top-k (it binds its input as the gate scores)
    and the declared pass died on the inner top-k with "no matcher recognised an expert block behind it" or on the
    stacked matcher's "the top-k neither feeds a softmax nor reads one" — no word of what the cluster needs; the
    matcher and reason asserts fail;
  * a router walk that took the group top-k's for routers lists them in `unfused_routers`; the count assert fails;
  * a walk that refused a plain softmax router (or a logits bias after the router mm) would stop a served model:
    the softmax-first block of the stacked test still fuses here.

Run: PYTHONPATH=src python -m pytest tests/unit/runtime/test_a_router_selecting_on_choice_scores_is_refused_by_name.py
"""
from __future__ import annotations

import copy

import pytest

from neurobrix.core.runtime.graph import moe_fusion as MF
from tests.unit.runtime.test_a_stacked_expert_block_routed_softmax_first_is_fused import qwen3vl_block
from neurobrix.pytorch.neurotax import moe_tokens

T, H, E, K = 7, 8, 16, 4


def _t(tid):
    return {"type": "tensor", "tensor_id": tid}


def _s(v):
    return {"type": "scalar", "value": v}


def _l(v):
    return {"type": "list", "value": v}


def grouped_router(n_group, topk_group, bias=True, sigmoid=True):
    """One layer's router in the recorded form of a sigmoid grouped router, small sizes, and a stacked expert slab
    read through the routing (indices, weights)."""
    tensors, ops, order = {}, {}, []

    def tensor(tid, shape, dt="float32", param=False):
        tensors[tid] = {"tensor_id": tid, "shape": list(shape), "dtype": dt, "is_parameter": param,
                        "weight_name": tid[len("param::"):] if param else None}
        return tid

    def op(uid, op_type, args, outs, kwargs=None):
        ops[uid] = {"op_uid": uid, "op_type": op_type,
                    "input_tensor_ids": [a["tensor_id"] for a in args if a.get("type") == "tensor"],
                    "output_tensor_ids": list(outs), "attributes": {"args": args, "kwargs": kwargs or {}},
                    "parent_module": "block.0.ffn"}
        order.append(uid)

    tensor("input::h", [T, H])
    w_r = tensor("param::block.0.ffn.gate.weight", [E, H], param=True)
    b_r = tensor("param::block.0.ffn.gate.choice_bias", [E], param=True)
    w_in = tensor("param::block.0.ffn.experts.gate_up_proj", [E, 2 * H, H], param=True)
    Eg = E // n_group
    op("t.r", "aten::t", [_t(w_r)], [tensor("tr", [H, E])])
    op("mm.r", "aten::mm", [_t("input::h"), _t("tr")], [tensor("logits", [T, E])])
    if sigmoid:
        op("sigmoid", "aten::sigmoid", [_t("logits")], [tensor("S", [T, E])])
    else:
        op("softmax", "aten::_softmax", [_t("logits"), _s(-1), _s(False)], [tensor("S", [T, E])])
    choice = "S"
    if bias:
        op("add.bias", "aten::add", [_t("S"), _t(b_r)], [tensor("C", [T, E])])
        choice = "C"
    op("view.g", "aten::view", [_t(choice), _l([-1, n_group, Eg])], [tensor("Cg", [T, n_group, Eg])])
    op("topk.in", "aten::topk", [_t("Cg"), _s(2), _s(-1)],
       [tensor("tv", [T, n_group, 2]), tensor("ti", [T, n_group, 2], "int64")])
    op("sum.g", "aten::sum", [_t("tv"), _l([-1])], [tensor("gs", [T, n_group])])
    op("topk.g", "aten::topk", [_t("gs"), _s(topk_group), _s(-1), _s(True), _s(False)],
       [tensor("gv", [T, topk_group]), tensor("gi", [T, topk_group], "int64")])
    op("zeros", "aten::zeros_like", [_t("gs")], [tensor("gm0", [T, n_group])])
    op("scatter", "aten::scatter_", [_t("gm0"), _s(1), _t("gi"), _s(1)], [tensor("gm", [T, n_group])])
    op("unsq", "aten::unsqueeze", [_t("gm"), _s(-1)], [tensor("gu", [T, n_group, 1])])
    op("expand", "aten::expand", [_t("gu"), _l([-1, n_group, Eg])], [tensor("ge", [T, n_group, Eg])])
    op("reshape", "aten::reshape", [_t("ge"), _l([-1, E])], [tensor("gr", [T, E])])
    op("bool", "aten::_to_copy", [_t("gr")], [tensor("gb", [T, E], "bool")],
       kwargs={"dtype": {"type": "dtype", "value": "torch.bool"}})
    op("not", "aten::bitwise_not", [_t("gb")], [tensor("nm", [T, E], "bool")])
    op("mask", "aten::masked_fill", [_t(choice), _t("nm"), _s(0.0)], [tensor("Cm", [T, E])])
    op("topk", "aten::topk", [_t("Cm"), _s(K), _s(-1), _s(True), _s(False)],
       [tensor("fv", [T, K]), tensor("fi", [T, K], "int64")])
    op("gather", "aten::gather", [_t("S"), _s(1), _t("fi")], [tensor("w", [T, K])])
    op("sum.w", "aten::sum", [_t("w"), _l([-1]), _s(True)], [tensor("ws", [T, 1])])
    op("add.eps", "aten::add", [_t("ws"), _s(1e-20)], [tensor("wse", [T, 1])])
    op("div", "aten::div", [_t("w"), _t("wse")], [tensor("wn", [T, K])])
    op("mul", "aten::mul", [_t("wn"), _s(2.5)], [tensor("rw", [T, K])])
    op("index", "aten::index", [_t(w_in), {"type": "tensor_tuple", "tensor_ids": ["fi"]}],
       [tensor("wsel", [T, K, 2 * H, H])])
    op("mul.out", "aten::mul", [_t("wsel"), _t("rw")], [tensor("out", [T, K, 2 * H, H])])
    return {"tensors": tensors, "ops": ops, "execution_order": order,
            "input_tensor_ids": ["input::h"], "output_tensor_ids": ["out"]}


GEOMETRIES = [(1, 1), (8, 4)]


@pytest.mark.parametrize("n_group,topk_group", GEOMETRIES)
def test_a_declared_grouped_router_is_refused_by_name(n_group, topk_group):
    g = grouped_router(n_group, topk_group)
    with pytest.raises(RuntimeError) as err:
        MF.detect_and_fuse_moe(g, "llm", declared=True, tokens=moe_tokens())
    msg = str(err.value)
    assert "1 router(s) of a declared MoE left unfused" in msg, msg
    assert "topk:" in msg and "topk.in" not in msg and "topk.g" not in msg, msg
    assert "choice scores" in msg and "'aten::sigmoid'" in msg and f"n_group {n_group}" in msg, msg
    assert "Missing: a routing op bound to this cluster" in msg, msg


@pytest.mark.parametrize("n_group,topk_group", GEOMETRIES)
def test_no_matcher_is_handed_a_router_selecting_on_choice_scores(n_group, topk_group, monkeypatch):
    seen = []
    for name in ("_fuse_one_moe_layer", "_fuse_one_stacked_layer"):
        real = getattr(MF, name)
        monkeypatch.setattr(MF, name, lambda *a, _r=real, _n=name, **k: seen.append((_n, a[6])) or _r(*a, **k))
    g = grouped_router(n_group, topk_group)
    before = copy.deepcopy(g)
    refusals = {}
    out = MF.detect_and_fuse_moe(g, "llm", refusals=refusals, tokens=moe_tokens())
    assert seen == [], f"a matcher was handed the router or its group top-k: {seen}"
    assert list(refusals) == ["topk"] and "choice scores" in refusals["topk"], refusals
    assert out == before, "an undeclared refusal leaves the graph as recorded"


@pytest.mark.parametrize("n_group,topk_group", GEOMETRIES)
def test_the_group_top_ks_are_part_of_the_router(n_group, topk_group):
    assert MF.unfused_routers(grouped_router(n_group, topk_group)) == ["topk"]


@pytest.mark.parametrize("bias,sigmoid,named", [(False, True, "'aten::sigmoid'"),
                                                (True, False, "a choice bias")])
def test_each_mark_of_choice_scores_is_refused(bias, sigmoid, named):
    refusals = {}
    MF.detect_and_fuse_moe(grouped_router(8, 4, bias=bias, sigmoid=sigmoid), "llm", refusals=refusals, tokens=moe_tokens())
    assert list(refusals) == ["topk"] and named in refusals["topk"], refusals


def test_a_router_selecting_on_its_softmax_scores_still_fuses():
    g = MF.detect_and_fuse_moe(qwen3vl_block(), "llm", declared=True, tokens=moe_tokens())
    assert [o["op_type"] for o in g["ops"].values()].count("custom::moe_fused") == 1
