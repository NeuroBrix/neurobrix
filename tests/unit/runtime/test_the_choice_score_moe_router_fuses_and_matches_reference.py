"""Choice-score MoE routing (sigmoid/softmax + e_score_correction_bias + group top-k
+ gather of unbiased scores + renorm + routed_scaling_factor):

  1. `moe_fusion.detect_and_fuse_moe` matches and fuses choice-score routers feeding
     stacked `_grouped_mm` expert slabs (reading score_func, choice_bias_tid, n_group,
     topk_group, group_topk_k, mask_fill_value, norm_eps, routed_scaling_factor from
     the graph with ZERO hardcode), and refuses unrecognised choice-score clusters
     loudly under ZERO FALLBACK;
  2. `compiled_sequence.choice_route_torch` (PyTorch branch) and
     `triton/moe.choice_route_triton` (Triton branch, ZERO Torch) match the unfused
     reference numerically for both `n_group == 1` and `n_group > 1` (`topk_group < n_group`),
     with choice bias, renormalisation, and routed_scaling_factor.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from neurobrix.core.runtime.graph.compiled_sequence import choice_route_torch
from neurobrix.core.runtime.graph.moe_fusion import detect_and_fuse_moe, unfused_routers
from neurobrix.pytorch.neurotax import moe_tokens


def _make_grouped_mm_choice_dag(
    *,
    T: int = 7,
    E: int = 16,
    K: int = 4,
    H: int = 32,
    F_dim: int = 24,
    n_group: int = 1,
    topk_group: int = 1,
    group_topk_k: int = 2,
    norm_topk_prob: bool = True,
    norm_eps: float = 1e-20,
    routed_scaling_factor: float = 1.8,
    include_bias: bool = True,
    consume_topk_vals: bool = False,
):
    """Build a synthetic DAG mirroring a choice-score router + stacked `_grouped_mm` MoE block."""
    G = n_group
    GS = E // G
    ops = {}
    tensors = {}
    order = []

    def add_tensor(tid, shape, dtype="float32", is_param=False):
        tensors[tid] = {
            "tensor_id": tid,
            "shape": list(shape),
            "dtype": dtype,
            "is_parameter": is_param,
        }

    def add_op(uid, op_type, inputs, outputs, out_shapes, out_dtypes=None, attrs=None, parent="block.1.ffn"):
        if out_dtypes is None:
            out_dtypes = ["float32"] * len(outputs)
        for ot, sh, dt in zip(outputs, out_shapes, out_dtypes):
            add_tensor(ot, sh, dt)
        a = dict(attrs or {})
        if "args" not in a:
            a["args"] = [{"type": "tensor", "tensor_id": t} for t in inputs]
        a.setdefault("kwargs", {})
        ops[uid] = {
            "op_uid": uid,
            "op_type": op_type,
            "input_tensor_ids": list(inputs),
            "output_tensor_ids": list(outputs),
            "input_shapes": [tensors[t]["shape"] for t in inputs],
            "output_shapes": [list(s) for s in out_shapes],
            "output_dtypes": list(out_dtypes),
            "attributes": a,
            "parent_module": parent,
        }
        order.append(uid)

    add_tensor("hidden_in", [1, T, H], "bfloat16")
    add_tensor("router_logits", [T, E], "float32")
    add_tensor("param::block.1.ffn.expert.ffn_gate_up", [E, 2 * F_dim, H], "bfloat16", is_param=True)
    add_tensor("param::block.1.ffn.expert.down", [E, H, F_dim], "bfloat16", is_param=True)
    if include_bias:
        add_tensor("param::block.1.ffn.router.e_score_correction_bias", [E], "float32", is_param=True)

    # Choice routing cluster
    add_op("aten.sigmoid::0", "aten::sigmoid", ["router_logits"], ["sig_out"], [[T, E]])
    cur_choice = "sig_out"
    if include_bias:
        add_op(
            "aten.add::0", "aten::add",
            ["sig_out", "param::block.1.ffn.router.e_score_correction_bias"],
            ["choice_bias_out"], [[T, E]],
        )
        cur_choice = "choice_bias_out"

    # Group mask cluster
    add_op("aten.view::0", "aten::view", [cur_choice], ["grp_view"], [[T, G, GS]],
           attrs={"shape": [-1, G, GS]})
    add_op("aten.topk::0", "aten::topk", ["grp_view"], ["grp_tk_v", "grp_tk_i"],
           [[T, G, group_topk_k], [T, G, group_topk_k]], ["float32", "int64"],
           attrs={"args": [{"type": "tensor", "tensor_id": "grp_view"}, {"type": "scalar", "value": group_topk_k}]})
    add_op("aten.sum::0", "aten::sum", ["grp_tk_v"], ["grp_sum"], [[T, G]],
           attrs={"args": [{"type": "tensor", "tensor_id": "grp_tk_v"}, {"type": "list", "value": [-1]}], "dim": [-1]})
    add_op("aten.topk::1", "aten::topk", ["grp_sum"], ["sel_g_v", "sel_g_i"],
           [[T, topk_group], [T, topk_group]], ["float32", "int64"],
           attrs={"args": [{"type": "tensor", "tensor_id": "grp_sum"},
                           {"type": "scalar", "value": topk_group},
                           {"type": "scalar", "value": -1},
                           {"type": "scalar", "value": True},
                           {"type": "scalar", "value": False}]})
    add_op("aten.zeros_like::0", "aten::zeros_like", ["grp_sum"], ["grp_zeros"], [[T, G]])
    add_op("aten.scatter::0", "aten::scatter", ["grp_zeros", "sel_g_i"], ["grp_mask"], [[T, G]],
           attrs={"args": [{"type": "tensor", "tensor_id": "grp_zeros"},
                           {"type": "scalar", "value": 1},
                           {"type": "tensor", "tensor_id": "sel_g_i"},
                           {"type": "scalar", "value": 1}]})
    add_op("aten.unsqueeze::0", "aten::unsqueeze", ["grp_mask"], ["grp_unsq"], [[T, G, 1]],
           attrs={"args": [{"type": "tensor", "tensor_id": "grp_mask"}, {"type": "scalar", "value": -1}], "dim": -1})
    add_op("aten.expand::0", "aten::expand", ["grp_unsq"], ["grp_exp"], [[T, G, GS]],
           attrs={"size": [-1, G, GS]})
    add_op("aten.view::1", "aten::view", ["grp_exp"], ["mask_2d"], [[T, E]], attrs={"shape": [T, E]})
    add_op("aten._to_copy::0", "aten::_to_copy", ["mask_2d"], ["mask_bool"], [[T, E]], ["bool"])
    add_op("aten.bitwise_not::0", "aten::bitwise_not", ["mask_bool"], ["inv_mask"], [[T, E]], ["bool"])
    add_op("aten.masked_fill::0", "aten::masked_fill", [cur_choice, "inv_mask"], ["masked_choice"], [[T, E]],
           attrs={"args": [{"type": "tensor", "tensor_id": cur_choice},
                           {"type": "tensor", "tensor_id": "inv_mask"},
                           {"type": "scalar", "value": 0.0}]})

    # Router top-k
    add_op("aten.topk::2", "aten::topk", ["masked_choice"], ["tk_vals", "tk_idx"],
           [[T, K], [T, K]], ["float32", "int64"],
           attrs={"args": [{"type": "tensor", "tensor_id": "masked_choice"},
                           {"type": "scalar", "value": K},
                           {"type": "scalar", "value": -1},
                           {"type": "scalar", "value": True},
                           {"type": "scalar", "value": False}]})

    if consume_topk_vals:
        add_op("aten.neg::bad", "aten::neg", ["tk_vals"], ["bad_out"], [[T, K]])

    # Gather unbiased scores
    add_op("aten.gather::0", "aten::gather", ["sig_out", "tk_idx"], ["raw_w"], [[T, K]],
           attrs={"args": [{"type": "tensor", "tensor_id": "sig_out"},
                           {"type": "scalar", "value": 1},
                           {"type": "tensor", "tensor_id": "tk_idx"}], "dim": 1})
    w_cur = "raw_w"
    if norm_topk_prob:
        add_op("aten.sum::1", "aten::sum", [w_cur], ["w_sum"], [[T, 1]],
               attrs={"args": [{"type": "tensor", "tensor_id": w_cur},
                               {"type": "list", "value": [-1]},
                               {"type": "scalar", "value": True}], "dim": [-1], "keepdim": True})
        denom = "w_sum"
        if norm_eps != 0.0:
            add_op("aten.add::1", "aten::add", ["w_sum"], ["w_denom"], [[T, 1]],
                   attrs={"args": [{"type": "tensor", "tensor_id": "w_sum"},
                                   {"type": "scalar", "value": norm_eps}]})
            denom = "w_denom"
        add_op("aten.div::0", "aten::div", [w_cur, denom], ["w_norm"], [[T, K]])
        w_cur = "w_norm"
    if routed_scaling_factor != 1.0:
        add_op("aten.mul::0", "aten::mul", [w_cur], ["w_scaled"], [[T, K]],
               attrs={"args": [{"type": "tensor", "tensor_id": w_cur},
                               {"type": "scalar", "value": routed_scaling_factor}]})
        w_cur = "w_scaled"

    # Stacked _grouped_mm dispatch
    TK = T * K
    add_op("aten.view::2", "aten::view", ["hidden_in"], ["h_2d"], [[T, H]], ["bfloat16"], parent="block.1.ffn")
    add_op("aten.view::3", "aten::view", ["tk_idx"], ["flat_idx"], [[TK]], ["int64"], parent="block.1.ffn.expert")
    add_op("aten.arange::0", "aten::arange", [], ["tok_ar"], [[T]], ["int64"], parent="block.1.ffn.expert")
    add_op("aten.unsqueeze::1", "aten::unsqueeze", ["tok_ar"], ["tok_u"], [[T, 1]], ["int64"], parent="block.1.ffn.expert")
    add_op("aten.expand::1", "aten::expand", ["tok_u"], ["tok_e"], [[T, K]], ["int64"], parent="block.1.ffn.expert")
    add_op("aten.clone::0", "aten::clone", ["tok_e"], ["tok_c"], [[T, K]], ["int64"], parent="block.1.ffn.expert")
    add_op("aten._unsafe_view::0", "aten::_unsafe_view", ["tok_c"], ["tok_flat"], [[TK]], ["int64"], parent="block.1.ffn.expert")
    add_op("aten.view::4", "aten::view", [w_cur], ["w_flat"], [[TK, 1]], ["float32"], parent="block.1.ffn.expert")
    add_op("aten.index::0", "aten::index", ["h_2d", "tok_flat"], ["h_exp"], [[TK, H]], ["bfloat16"], parent="block.1.ffn.expert")
    add_op("aten.sort::0", "aten::sort", ["flat_idx"], ["s_exp", "perm"], [[TK], [TK]], ["int64", "int64"], parent="block.1.ffn.expert")
    add_op("aten.sort::1", "aten::sort", ["perm"], ["s_perm", "inv_perm"], [[TK], [TK]], ["int64", "int64"], parent="block.1.ffn.expert")
    add_op("aten.index::1", "aten::index", ["flat_idx", "perm"], ["p_idx"], [[TK]], ["int64"], parent="block.1.ffn.expert")
    add_op("aten.index::2", "aten::index", ["w_flat", "perm"], ["p_w"], [[TK, 1]], ["float32"], parent="block.1.ffn.expert")
    add_op("aten.index::3", "aten::index", ["h_exp", "perm"], ["p_h"], [[TK, H]], ["bfloat16"], parent="block.1.ffn.expert")
    add_op("aten._to_copy::1", "aten::_to_copy", ["p_idx"], ["p_idx_f"], [[TK]], ["float32"], parent="block.1.ffn.expert")
    add_op("aten.histc::0", "aten::histc", ["p_idx_f"], ["counts"], [[E]], ["float32"], parent="block.1.ffn.expert")
    add_op("aten.cumsum::0", "aten::cumsum", ["counts"], ["offs"], [[E]], ["int32"], parent="block.1.ffn.expert")
    add_op("aten.transpose::0", "aten::transpose", ["param::block.1.ffn.expert.ffn_gate_up"], ["w_in_t"],
           [[E, H, 2 * F_dim]], ["bfloat16"], attrs={"dim0": -2, "dim1": -1}, parent="block.1.ffn.expert")
    add_op("aten._grouped_mm::0", "aten::_grouped_mm", ["p_h", "w_in_t", "offs"], ["gmm0_out"], [[TK, 2 * F_dim]], ["bfloat16"], parent="block.1.ffn.expert")
    add_op("aten.split::0", "aten::split", ["gmm0_out"], ["gate_h", "up_h"], [[TK, F_dim], [TK, F_dim]],
           ["bfloat16", "bfloat16"], attrs={"split_size": F_dim, "dim": -1}, parent="block.1.ffn.expert")
    add_op("aten.silu::0", "aten::silu", ["gate_h"], ["silu_h"], [[TK, F_dim]], ["bfloat16"], parent="block.1.ffn.expert")
    add_op("aten.mul::1", "aten::mul", ["silu_h", "up_h"], ["swiglu_h"], [[TK, F_dim]], ["bfloat16"], parent="block.1.ffn.expert")
    add_op("aten.transpose::1", "aten::transpose", ["param::block.1.ffn.expert.down"], ["w_out_t"],
           [[E, F_dim, H]], ["bfloat16"], attrs={"dim0": -2, "dim1": -1}, parent="block.1.ffn.expert")
    add_op("aten._grouped_mm::1", "aten::_grouped_mm", ["swiglu_h", "w_out_t", "offs"], ["gmm1_out"], [[TK, H]], ["bfloat16"], parent="block.1.ffn.expert")
    add_op("aten.mul::2", "aten::mul", ["gmm1_out", "p_w"], ["w_out_m"], [[TK, H]], ["float32"], parent="block.1.ffn.expert")
    add_op("aten.index::4", "aten::index", ["w_out_m", "inv_perm"], ["unperm_h"], [[TK, H]], ["float32"], parent="block.1.ffn.expert")
    add_op("aten.view::5", "aten::view", ["unperm_h"], ["unflat_h"], [[T, K, H]], ["float32"], parent="block.1.ffn.expert")
    add_op("aten.sum::2", "aten::sum", ["unflat_h"], ["sum_h"], [[T, H]], ["float32"], attrs={"dim": [1]}, parent="block.1.ffn.expert")
    add_op("aten._to_copy::2", "aten::_to_copy", ["sum_h"], ["cast_h"], [[T, H]], ["bfloat16"], parent="block.1.ffn.expert")
    add_op("aten.view::6", "aten::view", ["cast_h"], ["block_out"], [[1, T, H]], ["bfloat16"], parent="block.1.ffn")

    return {"ops": ops, "tensors": tensors, "execution_order": order}


def _reference_choice_route_torch(
    logits: torch.Tensor,
    top_k: int,
    *,
    score_func: str = "sigmoid",
    choice_bias: torch.Tensor | None = None,
    n_group: int = 1,
    topk_group: int = 1,
    group_topk_k: int = 2,
    mask_fill_value: float = 0.0,
    norm_topk_prob: bool = True,
    norm_eps: float = 1e-20,
    routed_scaling_factor: float = 1.8,
):
    """Exact unfused vendor math (as recorded in graph.json)."""
    x = logits.float().reshape(-1, logits.shape[-1])
    T, E = x.shape
    scores = torch.sigmoid(x) if score_func == "sigmoid" else F.softmax(x, dim=-1)
    choice = scores + choice_bias.float() if choice_bias is not None else scores
    grouped = choice.view(T, n_group, E // n_group)
    group_scores = grouped.topk(group_topk_k, dim=-1).values.sum(dim=-1)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, largest=True, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores).scatter(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(-1).expand(T, n_group, E // n_group).reshape(T, E)
    masked = choice.masked_fill(~score_mask.bool(), mask_fill_value)
    _, idx = torch.topk(masked, k=top_k, dim=-1, largest=True, sorted=False)
    w = scores.gather(1, idx)
    if norm_topk_prob:
        w = w / (w.sum(dim=-1, keepdim=True) + norm_eps)
    w = w * routed_scaling_factor
    return w, idx


@pytest.mark.parametrize("n_group,topk_group,norm,scale", [
    (1, 1, True, 1.8),
    (4, 2, True, 2.5),
    (4, 2, False, 1.0),
])
def test_choice_score_grouped_mm_fuses_and_reads_all_graph_attrs(n_group, topk_group, norm, scale):
    dag = _make_grouped_mm_choice_dag(
        n_group=n_group, topk_group=topk_group, norm_topk_prob=norm, routed_scaling_factor=scale,
    )
    fused = detect_and_fuse_moe(dag, "llm", norm_topk_prob=norm, declared=True, tokens=moe_tokens())
    assert unfused_routers(fused) == []
    assert fused["execution_order"] == ["moe_fused::block.1"]
    op = fused["ops"]["moe_fused::block.1"]
    assert op["op_type"] == "custom::moe_fused"
    attrs = op["attributes"]
    assert attrs["gate_scores_tid"] == "router_logits"
    assert attrs["hidden_states_tid"] == "hidden_in"
    assert op["output_tensor_ids"] == ["block_out"]
    assert "param::block.1.ffn.router.e_score_correction_bias" in op["input_tensor_ids"]
    assert attrs["norm_topk_prob"] is norm
    assert attrs["routing_from_graph"] == "choice_score"
    cr = attrs["choice_routing"]
    assert cr == {
        "score_func": "sigmoid",
        "choice_bias_tid": "param::block.1.ffn.router.e_score_correction_bias",
        "n_group": n_group,
        "topk_group": topk_group,
        "group_topk_k": 2,
        "mask_fill_value": 0.0,
        "norm_eps": 1e-20 if norm else 0.0,
        "routed_scaling_factor": scale,
    }
    st = attrs["stacked_experts"]
    assert st["ffn_dim"] == 24
    assert st["input_linear_in_axis"] == 1
    assert st["output_linear_in_axis"] == 1
    assert st["gate_offset"] == 0


def test_zero_fallback_refuses_unrecognised_choice_router_and_norm_mismatch():
    bad_dag = _make_grouped_mm_choice_dag(consume_topk_vals=True)
    with pytest.raises(RuntimeError, match="ZERO FALLBACK"):
        detect_and_fuse_moe(bad_dag, "llm", norm_topk_prob=True, declared=True, tokens=moe_tokens())

    mismatch_dag = _make_grouped_mm_choice_dag(norm_topk_prob=True)
    with pytest.raises(RuntimeError, match="ZERO FALLBACK"):
        detect_and_fuse_moe(mismatch_dag, "llm", norm_topk_prob=False, declared=True, tokens=moe_tokens())

    with pytest.raises(RuntimeError, match="ZERO FALLBACK"):
        choice_route_torch(torch.randn(2, 8), 2, True, {"score_func": "tanh"})

    from neurobrix.triton.moe import choice_route_triton
    with pytest.raises(RuntimeError, match="ZERO FALLBACK"):
        choice_route_triton(np.zeros((2, 8), dtype=np.float32), 2, True, {"score_func": "tanh"})


@pytest.mark.parametrize("score_func", ["sigmoid", "softmax"])
@pytest.mark.parametrize("n_group,topk_group,group_topk_k", [(1, 1, 2), (4, 2, 2), (8, 4, 2)])
@pytest.mark.parametrize("norm_topk_prob,norm_eps,scale", [(True, 1e-20, 1.8), (True, 0.0, 2.5), (False, 0.0, 1.0)])
def test_choice_routing_parity_torch_vs_triton_vs_unfused_reference(
    score_func, n_group, topk_group, group_topk_k, norm_topk_prob, norm_eps, scale
):
    from nbxtensor.tensor import NBXDtype, NBXTensor
    from neurobrix.triton.moe import choice_route_triton

    gen = torch.Generator().manual_seed(1234 + n_group * 17 + topk_group * 3)
    T, E, top_k = 11, 64, 4
    logits = torch.randn(T, E, generator=gen, dtype=torch.float32) * 1.5
    # Large choice bias deliberately flips selection vs unbiased scores
    bias = torch.randn(E, generator=gen, dtype=torch.float32) * 0.8

    spec = {
        "score_func": score_func,
        "choice_bias_tid": "bias",
        "n_group": n_group,
        "topk_group": topk_group,
        "group_topk_k": group_topk_k,
        "mask_fill_value": 0.0,
        "norm_eps": norm_eps,
        "routed_scaling_factor": scale,
    }

    w_ref, i_ref = _reference_choice_route_torch(
        logits, top_k, score_func=score_func, choice_bias=bias,
        n_group=n_group, topk_group=topk_group, group_topk_k=group_topk_k,
        norm_topk_prob=norm_topk_prob, norm_eps=norm_eps, routed_scaling_factor=scale,
    )
    w_pt, i_pt = choice_route_torch(logits, top_k, norm_topk_prob, spec, choice_bias=bias)

    # Triton branch (pure numpy AND CPU NBXTensor, ZERO Torch)
    w_tr_np, i_tr_np = choice_route_triton(logits.numpy(), top_k, norm_topk_prob, spec, choice_bias=bias.numpy())
    nbx_logits = NBXTensor.empty_cpu(logits.shape, dtype=NBXDtype.float32)
    nbx_logits._base[...] = logits.numpy()
    nbx_bias = NBXTensor.empty_cpu(bias.shape, dtype=NBXDtype.float32)
    nbx_bias._base[...] = bias.numpy()
    w_tr_nbx, i_tr_nbx = choice_route_triton(nbx_logits, top_k, norm_topk_prob, spec, choice_bias=nbx_bias)

    # Compare per-token (expert_id -> weight) mapping since sorted=False may permute ties/order
    def _as_map(w_arr, i_arr):
        return [{int(e): float(w) for e, w in zip(row_i, row_w)} for row_w, row_i in zip(w_arr, i_arr)]

    ref_m = _as_map(w_ref.numpy(), i_ref.numpy())
    pt_m = _as_map(w_pt.numpy(), i_pt.numpy())
    tr_np_m = _as_map(w_tr_np, i_tr_np)
    tr_nbx_m = _as_map(w_tr_nbx.numpy(), i_tr_nbx.numpy())

    for t in range(T):
        assert set(pt_m[t].keys()) == set(ref_m[t].keys()), f"PyTorch expert set mismatch at token {t}"
        assert set(tr_np_m[t].keys()) == set(ref_m[t].keys()), f"Triton np expert set mismatch at token {t}"
        assert set(tr_nbx_m[t].keys()) == set(ref_m[t].keys()), f"Triton NBX expert set mismatch at token {t}"
        for e in ref_m[t]:
            assert pt_m[t][e] == pytest.approx(ref_m[t][e], rel=1e-6, abs=1e-6)
            assert tr_np_m[t][e] == pytest.approx(ref_m[t][e], rel=1e-6, abs=1e-6)
            assert tr_nbx_m[t][e] == pytest.approx(ref_m[t][e], rel=1e-6, abs=1e-6)


def test_triton_execute_moe_fused_delegates_choice_routing_without_torch(monkeypatch):
    import neurobrix.triton.moe as M

    # Verify ZERO Torch import anywhere in neurobrix/triton/moe.py
    tree = ast.parse(Path(M.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] != "torch" for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] != "torch"

    seen = []
    monkeypatch.setattr(M, "fused_moe", lambda *a, **k: seen.append((a, k)) or "ok")
    logits = np.array([[0.2, -0.5, 1.3, 0.8, -1.0, 0.4, 0.1, 0.9]], dtype=np.float32)
    bias = np.array([0.0, 2.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    spec = {
        "score_func": "sigmoid",
        "choice_bias_tid": "b",
        "n_group": 2,
        "topk_group": 1,
        "group_topk_k": 2,
        "mask_fill_value": 0.0,
        "norm_eps": 1e-20,
        "routed_scaling_factor": 1.8,
    }
    res = M.execute_moe_fused(
        logits, "hidden", [], [], [], top_k=2, num_experts=8, norm_topk_prob=True,
        choice_routing=spec, choice_bias=bias,
    )
    assert res == "ok"
    (a, k), = seen
    assert a[0] is None, "gate_scores must be None when pre-routed by choice_route_triton"
    assert k["norm_topk_prob"] is False, "fused_moe must not double-normalise choice weights"
    assert k["topk_indices"].shape == (1, 2)
    assert k["topk_weights"].shape == (1, 2)
    assert float(k["topk_weights"].sum()) == pytest.approx(1.8, rel=1e-6, abs=1e-6)


def test_real_glm47_flash_graph_fuses_all_46_routers_when_present():
    gpath = Path("/mnt/nvme_ai_work/built_models_1127/GLM-4.7-Flash/GLM-4.7-Flash/components/model/graph.json")
    if not gpath.exists():
        pytest.skip("GLM-4.7-Flash container not mounted on this host")
    dag = json.loads(gpath.read_text())
    fused = detect_and_fuse_moe(dag, "llm", norm_topk_prob=True, declared=True, tokens=moe_tokens())
    assert unfused_routers(fused) == []
    moe_ops = [u for u in fused["execution_order"] if fused["ops"][u]["op_type"] == "custom::moe_fused"]
    assert len(moe_ops) == 46
    for u in moe_ops:
        attrs = fused["ops"][u]["attributes"]
        assert attrs["routing_from_graph"] == "choice_score"
        assert attrs["choice_routing"]["score_func"] == "sigmoid"
        assert attrs["choice_routing"]["n_group"] == 1
        assert attrs["choice_routing"]["topk_group"] == 1
        assert attrs["choice_routing"]["routed_scaling_factor"] == pytest.approx(1.8)
        assert attrs["num_experts"] == 64
        assert attrs["top_k"] == 4
        assert attrs["norm_topk_prob"] is True
