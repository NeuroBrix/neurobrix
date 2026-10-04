"""A stacked-expert MoE block routed softmax-first is fused, routed-only.

Qwen3-VL-30B-A3B-Thinking (transformers 4.57 `Qwen3VLMoeTextExperts`, inference
form) traces every MoE layer DENSE: softmax(logits) -> topk -> sum + div (the
renormalisation) -> cast -> scatter into zeros[T, E]; the hidden states repeated
once per expert -> bmm(W_in[E, H, 2F]) -> split(F) -> silu(gate) * up ->
bmm(W_out[E, F, H]) -> weighted by the routing matrix -> sum over the experts.
Correct output, but all E experts are read and computed for every token. Measured
2026-10-04 (nbx/campaigns/2026_10_04_moe_measure): 0 of 48 layers fused, 9.94x the
active bytes per decode token.

The stacked matcher now reads this block off the graph — the routing order (the
top-k READS a softmax), the renormalisation (the sum + div are there or not), the
slab geometry (which axis of W_in / W_out the bmm contracts), the gate half (the
split piece silu consumes) — and emits the SAME custom::moe_fused op the other
MoE models get. Its routing is already the fused op's own: topk of the softmax
scores, then the renormalisation, in fp32, cast to the weight dtype at the combine.

What these tests would do if the code were wrong:
  * a matcher that did not fuse, or bound the 2-D hidden view (the residual add
    would then read [T, H] for [B, S, H]), fails the structure asserts;
  * a reader that swapped the halves or read the slab along the wrong axis, or a
    routing that skipped / forced the renormalisation, makes the fused result
    differ from the unfused graph by O(1) — the oracle compares at fp64;
  * a matcher that fused a block with any other activation, a scaled routing
    weight or an escaping intermediate is caught by its declined-by-name cases.
Both engines' CPU reference paths run: the compiled closure (the compiled
mirror of the triton dispatch) and the PyTorch-sequential native op. The triton
grouped GEMM needs a card — its proof is the GPU token-for-token run.

Run: PYTHONPATH=src python -m pytest tests/unit/runtime/test_a_stacked_expert_block_routed_softmax_first_is_fused.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as TF

from neurobrix.core.paths import cache_dir
from neurobrix.core.runtime.graph import moe_fusion as MF

# ─────────────────────────────── the synthetic block ──────────────────────────────

B, S, H, F, E, K = 2, 5, 8, 3, 6, 2
T = B * S


def _t(tid):
    return {"type": "tensor", "tensor_id": tid}


def _s(v):
    return {"type": "scalar", "value": v}


def _l(v):
    return {"type": "list", "value": v}


def qwen3vl_block(renorm=True, gate_piece=0, act="aten::silu", scale_routing=False,
                  escape=False, idx_reused=False, dtype="float64"):
    """One MoE layer in the traced form of transformers 4.57's Qwen3-VL-MoE block
    (op for op as the container's graph.json, block.0), small sizes."""
    tensors, ops, order = {}, {}, []

    def tensor(tid, shape, dt=dtype, param=False):
        tensors[tid] = {"tensor_id": tid, "shape": list(shape), "dtype": dt,
                        "is_parameter": param,
                        "weight_name": tid[len("param::"):] if param else None}
        return tid

    def op(uid, op_type, args, outs, kwargs=None, parent="block.0.ffn"):
        ins = [a["tensor_id"] for a in args if a.get("type") == "tensor"]
        ops[uid] = {"op_uid": uid, "op_type": op_type, "input_tensor_ids": ins,
                    "output_tensor_ids": [o for o in outs],
                    "attributes": {"args": args, "kwargs": kwargs or {}},
                    "parent_module": parent}
        order.append(uid)

    tensor("input::h", [B, S, H])
    tensor("input::res", [B, S, H])
    w_r = tensor("param::block.0.ffn.router.weight", [E, H], param=True)
    w_in = tensor("param::block.0.ffn.expert.gate_up_proj", [E, H, 2 * F], param=True)
    w_out = tensor("param::block.0.ffn.expert.down", [E, F, H], param=True)

    op("view.a", "aten::view", [_t("input::h"), _l([-1, H])], [tensor("va", [T, H])])
    op("t.r", "aten::t", [_t(w_r)], [tensor("tr", [H, E])], parent="block.0.ffn.router")
    op("mm.r", "aten::mm", [_t("va"), _t("tr")], [tensor("logits", [T, E])],
       parent="block.0.ffn.router")
    op("cast.r", "aten::_to_copy", [_t("logits")], [tensor("l32", [T, E], "float32")],
       kwargs={"dtype": {"type": "dtype", "value": "torch.float32"}})
    op("softmax", "aten::_softmax", [_t("l32"), _s(-1), _s(False)],
       [tensor("probs", [T, E], "float32")])
    op("topk", "aten::topk", [_t("probs"), _s(K)],
       [tensor("scores", [T, K], "float32"), tensor("idx", [T, K], "int64")])
    w_tid = "scores"
    if renorm:
        op("sum.0", "aten::sum", [_t("scores"), _l([-1]), _s(True)],
           [tensor("ssum", [T, 1], "float32")])
        op("div.0", "aten::div", [_t("scores"), _t("ssum")], [tensor("sdiv", [T, K], "float32")])
        w_tid = "sdiv"
    if scale_routing:
        op("mul.scale", "aten::mul", [_t(w_tid), _s(2.5)], [tensor("sscaled", [T, K], "float32")])
        w_tid = "sscaled"
    op("cast.w", "aten::_to_copy", [_t(w_tid)], [tensor("wcast", [T, K])],
       kwargs={"dtype": {"type": "dtype", "value": f"torch.{dtype}"}})
    op("zeros", "aten::zeros_like", [_t("logits")], [tensor("z", [T, E])])
    op("scatter", "aten::scatter", [_t("z"), _s(1), _t("idx"), _t("wcast")],
       [tensor("R", [T, E])])
    if idx_reused:
        op("idx.view", "aten::view", [_t("idx"), _l([-1])], [tensor("idx_flat", [T * K], "int64")])
    op("view.b", "aten::view", [_t("va"), _l([B, -1, H])], [tensor("hb", [B, S, H])])
    op("view.c", "aten::view", [_t("hb"), _l([-1, H])], [tensor("hc", [T, H])],
       parent="block.0.ffn.expert")
    op("repeat", "aten::repeat", [_t("hc"), _l([E, 1])], [tensor("hrep", [E * T, H])],
       parent="block.0.ffn.expert")
    op("view.d", "aten::view", [_t("hrep"), _l([E, -1, H])], [tensor("hd", [E, T, H])],
       parent="block.0.ffn.expert")
    op("bmm.u", "aten::bmm", [_t("hd"), _t(w_in)], [tensor("gu", [E, T, 2 * F])],
       parent="block.0.ffn.expert")
    op("split", "aten::split", [_t("gu"), _s(F), _s(-1)],
       [tensor("p0", [E, T, F]), tensor("p1", [E, T, F])], parent="block.0.ffn.expert")
    g, u = ("p0", "p1") if gate_piece == 0 else ("p1", "p0")
    op("act", act, [_t(g)], [tensor("ag", [E, T, F])], parent="block.0.ffn.expert.act_fn")
    op("mul.a", "aten::mul", [_t(u), _t("ag")], [tensor("am", [E, T, F])],
       parent="block.0.ffn.expert")
    op("bmm.d", "aten::bmm", [_t("am"), _t(w_out)], [tensor("dn", [E, T, H])],
       parent="block.0.ffn.expert")
    op("view.e", "aten::view", [_t("dn"), _l([E, B, S, H])], [tensor("de", [E, B, S, H])],
       parent="block.0.ffn.expert")
    op("transpose", "aten::transpose", [_t("R"), _s(0), _s(1)], [tensor("Rt", [E, T])],
       parent="block.0.ffn.expert")
    op("view.f", "aten::view", [_t("Rt"), _l([E, B, -1])], [tensor("Rv", [E, B, S])],
       parent="block.0.ffn.expert")
    op("unsqueeze", "aten::unsqueeze", [_t("Rv"), _s(3)], [tensor("Ru", [E, B, S, 1])],
       parent="block.0.ffn.expert")
    op("mul.c", "aten::mul", [_t("de"), _t("Ru")], [tensor("wsum_in", [E, B, S, H])],
       parent="block.0.ffn.expert")
    op("sum.1", "aten::sum", [_t("wsum_in"), _l([0])], [tensor("moe_out", [B, S, H])],
       parent="block.0.ffn.expert")
    outs = ["out"]
    op("add", "aten::add", [_t("input::res"), _t("moe_out")], [tensor("out", [B, S, H])],
       parent="block.0")
    if escape:
        # an expert-chain intermediate read outside the block
        op("leak", "aten::mul", [_t("am"), _s(1.0)], [tensor("leaked", [E, T, F])],
           parent="block.0")
        outs.append("leaked")
    if idx_reused:
        outs.append("idx_flat")
    return {"tensors": tensors, "ops": ops, "execution_order": order,
            "input_tensor_ids": ["input::h", "input::res"], "output_tensor_ids": outs}


def _feeds(dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64).to(dtype)
    return {"input::h": rnd(B, S, H), "input::res": rnd(B, S, H),
            "param::block.0.ffn.router.weight": rnd(E, H),
            "param::block.0.ffn.expert.gate_up_proj": rnd(E, H, 2 * F) * 0.5,
            "param::block.0.ffn.expert.down": rnd(E, F, H) * 0.5}


# ────────────────────────── the unfused graph, executed ───────────────────────────

def _args(op, env):
    out = []
    for a in op["attributes"]["args"]:
        out.append(env[a["tensor_id"]] if a.get("type") == "tensor" else a.get("value"))
    return out


_DT = {"torch.float32": torch.float32, "torch.float64": torch.float64,
       "torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16}


def run_graph(dag, feeds, fused_engine=None):
    """Execute the DAG op by op in torch — the unfused graph's own math. A
    custom::moe_fused op is executed by `fused_engine(op_uid, op, env)`."""
    env = dict(feeds)
    for uid in dag["execution_order"]:
        op = dag["ops"][uid]
        t = op["op_type"]
        a = _args(op, env)
        if t == "custom::moe_fused":
            r = fused_engine(uid, op, env)
        elif t in ("aten::view", "aten::reshape", "aten::_unsafe_view"):
            r = a[0].reshape(a[1])
        elif t == "aten::t":
            r = a[0].t()
        elif t == "aten::mm":
            r = a[0] @ a[1]
        elif t == "aten::bmm":
            r = torch.bmm(a[0], a[1])
        elif t == "aten::_to_copy":
            r = a[0].to(_DT[op["attributes"]["kwargs"]["dtype"]["value"]])
        elif t == "aten::_softmax":
            r = torch.softmax(a[0], dim=a[1])
        elif t == "aten::topk":
            r = torch.topk(a[0], a[1], dim=-1)
        elif t == "aten::sum":
            r = a[0].sum(dim=a[1], keepdim=bool(a[2]) if len(a) > 2 else False)
        elif t == "aten::div":
            r = a[0] / a[1]
        elif t == "aten::mul":
            r = a[0] * a[1]
        elif t == "aten::add":
            r = a[0] + a[1]
        elif t == "aten::zeros_like":
            r = torch.zeros_like(a[0])
        elif t == "aten::scatter":
            r = a[0].scatter(a[1], a[2], a[3])
        elif t == "aten::repeat":
            r = a[0].repeat(*a[1])
        elif t == "aten::split":
            r = torch.split(a[0], a[1], dim=a[2])
        elif t == "aten::silu":
            r = TF.silu(a[0])
        elif t == "aten::gelu":
            r = TF.gelu(a[0])
        elif t == "aten::transpose":
            r = a[0].transpose(a[1], a[2])
        elif t == "aten::unsqueeze":
            r = a[0].unsqueeze(a[1])
        else:
            raise AssertionError(f"interpreter has no {t}")
        outs = op["output_tensor_ids"]
        if isinstance(r, (tuple, list)):
            for o, v in zip(outs, r):
                env[o] = v
        else:
            env[outs[0]] = r
    return env


def compiled_engine(uid, op, env):
    """The compiled engine's own dispatch closure (CompiledSequence), on CPU."""
    from neurobrix.core.runtime.graph.compiled_sequence import CompiledSequence
    cs = CompiledSequence.__new__(CompiledSequence)
    cs._tensor_id_to_slot, cs._slot_to_tensor_id = {}, {}
    cs._next_slot, cs._num_intermediates = 0, 0
    for tid in op["input_tensor_ids"]:
        cs._tensor_id_to_slot[tid] = cs._next_slot
        cs._slot_to_tensor_id[cs._next_slot] = tid
        cs._next_slot += 1
    cop = cs._compile_moe_fused_op(uid, op, ())
    arena = [None] * cs._next_slot
    for tid, slot in cs._tensor_id_to_slot.items():
        arena[slot] = env.get(tid)
    return cop.func(arena)


def native_engine(uid, op, env):
    """The PyTorch-sequential engine's native fused op, on CPU."""
    from neurobrix.core.runtime.graph_executor import GraphExecutor
    ex = GraphExecutor.__new__(GraphExecutor)
    ex._ctx = SimpleNamespace(tensor_store=dict(env), tensors_metadata=_native_meta[0],
                              weights={})
    return ex._execute_moe_fused_native(op)


_native_meta = [None]


def _fuse(dag, refusals=None):
    d = copy.deepcopy(dag)
    return MF.detect_and_fuse_moe(d, "multimodal", norm_topk_prob=True, declared=True,
                                  refusals=refusals)


def _fused_ops(dag):
    return [u for u in dag["execution_order"]
            if dag["ops"][u].get("op_type") == "custom::moe_fused"]


# ─────────────────────────────── (a) the matcher ──────────────────────────────────

def test_the_softmax_first_block_fuses_into_the_one_fused_op():
    dag = qwen3vl_block()
    refusals = {}
    d2 = _fuse(dag, refusals)
    fused = _fused_ops(d2)
    assert fused == ["moe_fused::block.0"], (fused, refusals)
    a = d2["ops"][fused[0]]["attributes"]
    assert a["gate_scores_tid"] == "probs", "the fused op reads the graph's softmax output"
    assert a["hidden_states_tid"] == "hb", (
        "hidden must be the [B, S, H] view the block output is shaped like, "
        f"got {a['hidden_states_tid']}")
    assert d2["ops"][fused[0]]["output_tensor_ids"] == ["moe_out"]
    st = a["stacked_experts"]
    assert (st["input_linear_tid"], st["output_linear_tid"]) == (
        "param::block.0.ffn.expert.gate_up_proj", "param::block.0.ffn.expert.down")
    assert (st["ffn_dim"], st["input_linear_in_axis"], st["gate_offset"],
            st["output_linear_in_axis"]) == (F, 0, 0, 0), st
    assert (a["top_k"], a["num_experts"], a["norm_topk_prob"]) == (K, E, True)
    assert a.get("routing_from_graph") == "softmax_before_topk"
    # nothing of the dense block survives; the router and the softmax do
    kinds = [d2["ops"][u]["op_type"] for u in d2["execution_order"]]
    for gone in ("aten::bmm", "aten::repeat", "aten::scatter", "aten::topk", "aten::split"):
        assert gone not in kinds, f"{gone} survived the rewrite"
    assert "aten::_softmax" in kinds and "aten::mm" in kinds
    # producers before consumers
    pos = {u: i for i, u in enumerate(d2["execution_order"])}
    prod = {o: u for u in d2["execution_order"] for o in d2["ops"][u]["output_tensor_ids"]}
    for u in d2["execution_order"]:
        for tid in d2["ops"][u]["input_tensor_ids"]:
            assert prod.get(tid) is None or pos[prod[tid]] < pos[u], (u, tid)


def test_the_gate_half_is_the_piece_silu_reads():
    d2 = _fuse(qwen3vl_block(gate_piece=1))
    st = d2["ops"]["moe_fused::block.0"]["attributes"]["stacked_experts"]
    assert st["gate_offset"] == F, st


def test_the_renormalisation_is_read_off_the_graph():
    d2 = _fuse(qwen3vl_block(renorm=False))
    assert d2["ops"]["moe_fused::block.0"]["attributes"]["norm_topk_prob"] is False


@pytest.mark.parametrize("variant, reason", [
    (dict(act="aten::gelu"), "silu(gate) * up"),
    (dict(scale_routing=True), "not scattered"),
    (dict(escape=True), "escapes the block"),
    (dict(idx_reused=True), "read beyond the routing scatter"),
])
def test_a_block_that_is_not_this_one_is_declined_by_name(variant, reason):
    refusals = {}
    d2 = _fuse(qwen3vl_block(**variant), refusals)
    assert not _fused_ops(d2), f"{variant} must not fuse"
    assert "topk" in refusals and reason in refusals["topk"], refusals
    # the declined graph is left exactly as traced
    assert d2["execution_order"] == qwen3vl_block(**variant)["execution_order"]


# ───────────────────── (b) the fused op computes the graph's math ─────────────────

@pytest.mark.parametrize("renorm", [True, False])
@pytest.mark.parametrize("gate_piece", [0, 1])
@pytest.mark.parametrize("engine", ["compiled", "native"])
def test_the_fused_op_equals_the_unfused_graph(renorm, gate_piece, engine):
    dag = qwen3vl_block(renorm=renorm, gate_piece=gate_piece)
    d2 = _fuse(dag)
    assert _fused_ops(d2)
    _native_meta[0] = d2["tensors"]
    run = compiled_engine if engine == "compiled" else native_engine
    for seed in range(4):
        feeds = _feeds(seed=seed)
        ref = run_graph(dag, feeds)["out"]
        got = run_graph(d2, feeds, fused_engine=run)["out"]
        assert got.shape == ref.shape, (got.shape, ref.shape)
        err = (got - ref).abs().max().item()
        assert err < 1e-12, (
            f"{engine} renorm={renorm} gate_piece={gate_piece} seed={seed}: "
            f"max |fused - graph| = {err:.3e}")


def test_the_fused_op_in_bf16_stays_within_the_combine_rounding():
    """bf16 as the container stores it: routing fp32 in both, the combine
    re-associated (graph: sum over E; fused: index_add over routed experts)."""
    dag = qwen3vl_block(dtype="bfloat16")
    d2 = _fuse(dag)
    _native_meta[0] = d2["tensors"]
    feeds = _feeds(dtype=torch.bfloat16, seed=7)
    ref = run_graph(dag, feeds)["out"].double()
    for run in (compiled_engine, native_engine):
        got = run_graph(d2, feeds, fused_engine=run)["out"].double()
        rel = ((got - ref).abs().max() / ref.abs().max()).item()
        assert rel < 2e-2, rel


# ─────────────────────── (c) the real containers, the engine's pass ───────────────

def _graph(model, comp):
    p = Path(cache_dir()) / model / "components" / comp / "graph.json"
    if not p.exists():
        pytest.skip(f"{model} is not in this machine's cache — proven where the trace lives")
    return json.loads(p.read_text()), p.parents[2]


def test_every_qwen3_vl_moe_layer_now_fuses():
    dag, root = _graph("Qwen3-VL-30B-A3B-Thinking", "model.language_model")
    lm = json.loads((root / "runtime" / "defaults.json").read_text())["lm_config"]
    routers = sum(1 for o in dag["ops"].values() if o.get("op_type") == "aten::topk")
    bmm_before = sum(1 for u in dag["execution_order"] if dag["ops"][u]["op_type"] == "aten::bmm")
    refusals = {}
    d2 = MF.detect_and_fuse_moe(dag, "multimodal", norm_topk_prob=lm["norm_topk_prob"],
                                declared=True, refusals=refusals)
    fused = _fused_ops(d2)
    assert routers == 48 and len(fused) == 48, (routers, len(fused), list(refusals.items())[:2])
    assert not refusals
    T_ = d2["tensors"]
    for u in fused:
        o = d2["ops"][u]
        a = o["attributes"]
        st = a["stacked_experts"]
        assert (a["num_experts"], a["top_k"]) == (lm["num_experts"], lm["num_experts_per_tok"])
        assert st["ffn_dim"] == lm["moe_intermediate_size"]
        assert (st["input_linear_in_axis"], st["gate_offset"], st["output_linear_in_axis"]) \
            == (0, 0, 0), st
        # the renormalisation the graph computes is the one the config declares
        assert a["norm_topk_prob"] is bool(lm["norm_topk_prob"])
        assert T_[a["hidden_states_tid"]]["shape"] == T_[o["output_tensor_ids"][0]]["shape"]
        assert d2["ops"].get(a["gate_scores_tid"]) is None and \
            T_[a["gate_scores_tid"]]["dtype"] == "float32"
    # no op left reads an expert slab; the two expert bmm of each layer are gone
    # (what remains is the rotary embedding's own)
    slabs = {d2["ops"][u]["attributes"]["stacked_experts"][k]
             for u in fused for k in ("input_linear_tid", "output_linear_tid")}
    for u in d2["execution_order"]:
        o = d2["ops"][u]
        if o["op_type"] == "custom::moe_fused":
            continue
        assert not slabs & set(o.get("input_tensor_ids", [])), (u, "reads a slab")
    bmm_after = sum(1 for u in d2["execution_order"] if d2["ops"][u]["op_type"] == "aten::bmm")
    assert bmm_before - bmm_after == 2 * 48, (bmm_before, bmm_after)
    pos = {u: i for i, u in enumerate(d2["execution_order"])}
    prod = {t: u for u in d2["execution_order"] for t in d2["ops"][u]["output_tensor_ids"]}
    for u in d2["execution_order"]:
        for tid in MF._collect_input_tids(d2["ops"][u]):
            assert prod.get(tid) is None or pos[prod[tid]] < pos[u], (u, tid)


@pytest.mark.parametrize("model, comp, family, layers", [
    ("granite-3.1-1b-a400m-instruct", "model", "llm", 24),
    ("deepseek-moe-16b-chat", "model", "llm", 27),
])
def test_the_other_moe_containers_fuse_as_before(model, comp, family, layers):
    dag, _ = _graph(model, comp)
    d2 = MF.detect_and_fuse_moe(dag, family, norm_topk_prob=True)
    assert len(_fused_ops(d2)) == layers
