"""`derived_census.runtime_graph` forwards `moe_tokens()` to both `detect_and_fuse_moe` passes
(family == "llm" and declared non-llm MoE LM), and `derived_census.shadow_env` preserves any
existing `PYTHONPATH` instead of overwriting it when running from a patched clone.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))


def _minimal_softmax_moe_dag() -> dict:
    """Minimal self-contained softmax-first stacked-expert MoE DAG (CPU-only pure dict)."""
    B, S, H, F_dim, E, K = 1, 4, 16, 8, 8, 2
    T = B * S
    tensors, ops, order = {}, {}, []

    def t(tid):
        return {"type": "tensor", "tensor_id": tid}

    def s(v):
        return {"type": "scalar", "value": v}

    def l(v):
        return {"type": "list", "value": v}

    def add_tensor(tid, shape, dtype="float32", is_param=False):
        tensors[tid] = {
            "tensor_id": tid,
            "shape": list(shape),
            "dtype": dtype,
            "is_parameter": is_param,
            "weight_name": tid[len("param::"):] if is_param else None,
        }
        return tid

    def add_op(uid, op_type, args, outs, kwargs=None, parent="block.0.ffn"):
        ins = [a["tensor_id"] for a in args if a.get("type") == "tensor"]
        ops[uid] = {
            "op_uid": uid,
            "op_type": op_type,
            "input_tensor_ids": ins,
            "output_tensor_ids": list(outs),
            "input_shapes": [tensors[x]["shape"] for x in ins],
            "output_shapes": [tensors[x]["shape"] for x in outs],
            "output_dtypes": [tensors[x]["dtype"] for x in outs],
            "attributes": {"args": args, "kwargs": kwargs or {}},
            "parent_module": parent,
        }
        order.append(uid)

    add_tensor("input::h", [B, S, H])
    add_tensor("router_logits", [T, E])
    w_in = add_tensor("param::block.0.ffn.expert.gate_up_proj", [E, H, 2 * F_dim], is_param=True)
    w_out = add_tensor("param::block.0.ffn.expert.down", [E, F_dim, H], is_param=True)

    add_op("softmax", "aten::_softmax", [t("router_logits"), s(-1), s(False)],
           [add_tensor("probs", [T, E])])
    add_op("topk", "aten::topk", [t("probs"), s(K)],
           [add_tensor("scores", [T, K]), add_tensor("idx", [T, K], "int64")])
    add_op("sum.0", "aten::sum", [t("scores"), l([-1]), s(True)],
           [add_tensor("ssum", [T, 1])])
    add_op("div.0", "aten::div", [t("scores"), t("ssum")],
           [add_tensor("sdiv", [T, K])])
    add_op("zeros", "aten::zeros_like", [t("router_logits")],
           [add_tensor("z", [T, E])])
    add_op("scatter", "aten::scatter", [t("z"), s(1), t("idx"), t("sdiv")],
           [add_tensor("R", [T, E])])

    add_op("view.c", "aten::view", [t("input::h"), l([-1, H])],
           [add_tensor("hc", [T, H])], parent="block.0.ffn.expert")
    add_op("repeat", "aten::repeat", [t("hc"), l([E, 1])],
           [add_tensor("hrep", [E * T, H])], parent="block.0.ffn.expert")
    add_op("view.d", "aten::view", [t("hrep"), l([E, -1, H])],
           [add_tensor("hd", [E, T, H])], parent="block.0.ffn.expert")
    add_op("bmm.u", "aten::bmm", [t("hd"), t(w_in)],
           [add_tensor("gu", [E, T, 2 * F_dim])], parent="block.0.ffn.expert")
    add_op("split", "aten::split", [t("gu"), s(F_dim), s(-1)],
           [add_tensor("p0", [E, T, F_dim]), add_tensor("p1", [E, T, F_dim])],
           parent="block.0.ffn.expert")
    add_op("act", "aten::silu", [t("p0")],
           [add_tensor("ag", [E, T, F_dim])], parent="block.0.ffn.expert.act_fn")
    add_op("mul.a", "aten::mul", [t("p1"), t("ag")],
           [add_tensor("am", [E, T, F_dim])], parent="block.0.ffn.expert")
    add_op("bmm.d", "aten::bmm", [t("am"), t(w_out)],
           [add_tensor("dn", [E, T, H])], parent="block.0.ffn.expert")
    add_op("view.e", "aten::view", [t("dn"), l([E, B, S, H])],
           [add_tensor("de", [E, B, S, H])], parent="block.0.ffn.expert")

    add_op("transpose", "aten::transpose", [t("R"), s(0), s(1)],
           [add_tensor("Rt", [E, T])], parent="block.0.ffn.expert")
    add_op("view.f", "aten::view", [t("Rt"), l([E, B, -1])],
           [add_tensor("Rv", [E, B, S])], parent="block.0.ffn.expert")
    add_op("unsqueeze", "aten::unsqueeze", [t("Rv"), s(3)],
           [add_tensor("Ru", [E, B, S, 1])], parent="block.0.ffn.expert")
    add_op("mul.c", "aten::mul", [t("de"), t("Ru")],
           [add_tensor("wsum_in", [E, B, S, H])], parent="block.0.ffn.expert")
    add_op("sum.1", "aten::sum", [t("wsum_in"), l([0])],
           [add_tensor("moe_out", [B, S, H])], parent="block.0.ffn.expert")

    return {
        "ops": ops,
        "tensors": tensors,
        "execution_order": order,
        "input_tensor_ids": ["input::h", "router_logits"],
        "output_tensor_ids": ["moe_out"],
    }


def _write_fake_model(root: Path, model: str, family: str, comp: str = "lm",
                      *, declared_moe: bool = False, norm_topk_prob: bool = True) -> None:
    mdir = root / model
    (mdir / "components" / comp).mkdir(parents=True, exist_ok=True)
    (mdir / "runtime").mkdir(parents=True, exist_ok=True)
    (mdir / "manifest.json").write_text(json.dumps({"family": family}))
    (mdir / "components" / comp / "graph.json").write_text(json.dumps(_minimal_softmax_moe_dag()))
    if declared_moe:
        (mdir / "topology.json").write_text(json.dumps({
            "components": {comp: {}},
            "flow": {"type": "vlm", "vlm": {"lm_component": comp}},
        }))
        (mdir / "runtime" / "defaults.json").write_text(json.dumps({
            "lm_config": {"num_experts": 8, "norm_topk_prob": norm_topk_prob},
        }))
    else:
        (mdir / "topology.json").write_text(json.dumps({"components": {comp: {}}, "flow": {}}))


@pytest.mark.parametrize("family,declared_moe,expected_calls", [
    ("llm", False, [False]),
    ("multimodal", True, [False, True]),
])
def test_runtime_graph_passes_moe_tokens_to_detect_and_fuse_moe(
    tmp_path, monkeypatch, family, declared_moe, expected_calls
):
    import derived_census as D
    from neurobrix.core.runtime.graph import moe_fusion
    from neurobrix.triton.neurotax import moe_tokens

    _write_fake_model(tmp_path, "FakeMoE", family, comp="lm", declared_moe=declared_moe)
    monkeypatch.setattr(D, "CACHE", tmp_path)
    D._RUNTIME_GRAPHS.clear()

    seen_calls = []
    real_detect = moe_fusion.detect_and_fuse_moe

    def spy_detect(dag, fam, norm_topk_prob=True, declared=False, refusals=None, *, tokens):
        seen_calls.append({"family": fam, "declared": declared, "tokens": tokens})
        return real_detect(dag, fam, norm_topk_prob=norm_topk_prob, declared=declared,
                           refusals=refusals, tokens=tokens)

    monkeypatch.setattr(moe_fusion, "detect_and_fuse_moe", spy_detect)

    try:
        g = D.runtime_graph("FakeMoE", "lm")
        assert moe_fusion.unfused_routers(g) == []
        assert [g["ops"][u]["op_type"] for u in g["execution_order"]] == [
            "aten::_softmax",
            "custom::moe_fused",
        ]
        assert [c["declared"] for c in seen_calls] == expected_calls
        expected_tokens = moe_tokens()
        for c in seen_calls:
            assert c["tokens"] == expected_tokens
    finally:
        D._RUNTIME_GRAPHS.clear()


def test_shadow_env_preserves_pythonpath(monkeypatch):
    import derived_census as D

    monkeypatch.setenv("PYTHONPATH", "/existing/one:/existing/two")
    env = D.shadow_env("/patched/src", rung=16384)
    assert env["PYTHONPATH"] == f"/patched/src{os.pathsep}/existing/one:/existing/two"
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["NBX_CENSUS"] == "1"
    assert env["NBX_CENSUS_DEVICES"] == "1"
    assert env["NBX_PRISM_BUDGET_MB"] == "16384"

    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("NBX_PRISM_BUDGET_MB", raising=False)
    env_bare = D.shadow_env("/patched/src")
    assert env_bare["PYTHONPATH"] == "/patched/src"
    assert "NBX_PRISM_BUDGET_MB" not in env_bare
