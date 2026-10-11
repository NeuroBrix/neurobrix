"""A routed (MoE) component's topology is READ from the container, and its working set is STATED.

The plan priced a 30B-A3B model on a 32 GB card as 57 GB dense and streamed it whole per token,
although each routed layer reads 8 of its 128 experts. Sparsity is a dimension of the plan: the
topology (experts, top_k, shared experts, bytes per expert) is read from the graph as the engines
execute it — the declared-MoE fusion's `custom::moe_fused` ops — and the routed working set is
stated beside the plan's figures, never read to place (`core/prism/expert_topology.py`).

Two parts:

* the catalogue's routed containers, read from the shared cache (read-only), each component's
  topology asserted against the VENDOR's config.json — the snapshot when this machine has one,
  else the values copied from the vendor's repository with its URL. A value the reader got wrong
  (an expert's bytes, its top_k, its layer count, its shared expert) turns a row red;
* the pricing law on synthetic topologies, no container needed.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from neurobrix.core.paths import cache_dir
from neurobrix.core.prism.expert_topology import (
    ExpertTopology, RoutedLayer, price_working_set, read_expert_topology,
)
from neurobrix.core.prism.solver import PrismSolver

MB = 1024 * 1024
SNAPSHOTS = Path.home() / "hf_snapshots"

#: Qwen/Qwen3-30B-A3B-Thinking-2507 has no snapshot on this machine: its config.json, as served at
#: https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507/blob/main/config.json (read 2026-10-11),
#: the MoE keys only.
_QWEN3_30B_THINKING_2507 = {
    "hidden_size": 2048, "num_hidden_layers": 48, "num_experts": 128, "num_experts_per_tok": 8,
    "moe_intermediate_size": 768, "norm_topk_prob": True, "decoder_sparse_step": 1, "mlp_only_layers": [],
}

#: model -> {component: (where the vendor config lives, the path of the text config inside it)}.
CATALOGUE = {
    "deepseek-moe-16b-chat": {"model": ("deepseek-moe-16b-chat", ())},
    "granite-3.1-1b-a400m-instruct": {"model": ("granite-3.1-1b-a400m-instruct", ())},
    "Qwen3-30B-A3B-Thinking-2507": {"model": (_QWEN3_30B_THINKING_2507, ())},
    "Qwen3-Coder-30B-A3B-Instruct": {"model": ("Qwen3-Coder-30B-A3B-Instruct", ())},
    "Qwen3-Omni-30B-A3B-Instruct": {
        "thinker.model": ("Qwen3-Omni-30B-A3B-Instruct", ("thinker_config", "text_config")),
        "talker.model": ("Qwen3-Omni-30B-A3B-Instruct", ("talker_config", "text_config"))},
    "Qwen3-VL-30B-A3B-Thinking": {"model.language_model": ("Qwen3-VL-30B-A3B-Thinking", ("text_config",))},
}


def _vendor(source, path):
    if isinstance(source, dict):
        cfg = source
    else:
        p = SNAPSHOTS / source / "config.json"
        if not p.exists():
            pytest.skip(f"no vendor config.json for {source} on this machine")
        cfg = json.loads(p.read_text())
    for k in path:
        cfg = cfg[k]
    return cfg


def _expected(v):
    """The topology the vendor's config states, in the vendors' own spellings."""
    n = next(v[k] for k in ("n_routed_experts", "num_experts", "num_local_experts") if k in v)
    width = v.get("moe_intermediate_size", v.get("intermediate_size"))
    if v.get("n_shared_experts"):
        shared_width = int(v["n_shared_experts"]) * int(v["moe_intermediate_size"])
    else:
        shared_width = int(v.get("shared_expert_intermediate_size") or 0)
    assert int(v.get("decoder_sparse_step", 1)) == 1 and not v.get("mlp_only_layers") \
        and int(v.get("moe_layer_freq", 1)) == 1, "a sparse-step layout this table does not model"
    layers = int(v["num_hidden_layers"]) - int(v.get("first_k_dense_replace", 0))
    h = int(v["hidden_size"])
    return {"n_experts": int(n), "top_k": int(v["num_experts_per_tok"]), "routed_layers": layers,
            # gate, up and down: three H x F matrices, bf16 as every one of these is stored
            "expert_bytes": 3 * h * int(width) * 2, "shared_bytes": 3 * h * shared_width * 2,
            "norm_topk_prob": v.get("norm_topk_prob")}


def _topology(model, component):
    root = cache_dir() / model
    if not (root / "components" / component / "graph.json").exists():
        pytest.skip(f"{model}/{component} is not in this machine's cache")
    graph = json.loads((root / "components" / component / "graph.json").read_text())
    index = json.loads((root / "components" / component / "weights_index.json").read_text())
    container = SimpleNamespace(cache_path=root)
    solver = PrismSolver.__new__(PrismSolver)
    # The plan's own path: the graph as executed (the declared-MoE fusion on a copy), then the reader.
    executed = solver._graph_as_executed(SimpleNamespace(graph=graph, weights_index=index, name=component),
                                         container)
    lm = solver._read_lm_config(container)
    declared = (lm or {}).get("component_name") or (solver._read_defaults(container) or {}).get("lm_component")
    return read_expert_topology(component, executed.graph, executed.weights_index, 1.0, lm, declared), lm, declared


@pytest.mark.parametrize("model,component", [(m, c) for m, comps in CATALOGUE.items() for c in comps])
def test_the_topology_read_from_the_container_is_the_vendors(model, component):
    source, path = CATALOGUE[model][component]
    want = _expected(_vendor(source, path))
    topo, lm, declared = _topology(model, component)
    assert topo is not None, f"{model}/{component}: no routed layer read"
    assert len(topo.layers) == want["routed_layers"]
    assert topo.n_experts == [want["n_experts"]]
    assert topo.top_k == [want["top_k"]]
    assert {l.expert_bytes for l in topo.layers} == {want["expert_bytes"]}
    assert {l.shared_bytes for l in topo.layers} == {want["shared_bytes"]}
    if want["norm_topk_prob"] is not None and component == declared:
        assert {l.norm_topk_prob for l in topo.layers} == {bool(want["norm_topk_prob"])}
    if component == declared and want["shared_bytes"] == 0:
        assert topo.shared_experts == 0
    v = _vendor(source, path)
    if v.get("n_shared_experts"):
        assert topo.shared_experts == int(v["n_shared_experts"])


def _layer(e, k, b, shared=0):
    return RoutedLayer(op_uid="u", block=0, n_experts=e, top_k=k, norm_topk_prob=True,
                       expert_bytes=b, shared_bytes=shared)


def test_the_working_set_is_dense_plus_k_experts_plus_a_cache_on_the_rung():
    t = ExpertTopology("lm", [_layer(128, 8, 9 * MB) for _ in range(48)], shared_experts=0)
    weights = {"lm": 2048 * MB + t.routed_bytes, "head": 594 * MB}
    r = price_working_set({"lm": t}, weights, 32768, 4096 * MB, "a 32 GB rung")
    assert r["dense_mb"] == 2048 + 594
    assert r["active_experts_mb"] == 48 * 8 * 9
    assert r["minimum_mb"] == 2642 + 3456
    assert r["expert_cache_mb"] == 32768 - 6098 - 4096
    assert r["working_set_mb"] == 32768 - 4096
    assert r["cached_experts_per_layer"] == int((32768 - 6098 - 4096) // (48 * 9))
    assert r["dense_equivalent_mb"] == 2642 + 48 * 128 * 9


def test_the_cache_never_exceeds_the_idle_experts_nor_goes_negative():
    small = ExpertTopology("lm", [_layer(32, 8, MB) for _ in range(24)])
    r = price_working_set({"lm": small}, {"lm": 100 * MB + small.routed_bytes}, 32768, 0, "rung")
    assert r["expert_cache_mb"] == 24 * 24          # every idle expert, no more
    assert r["cached_experts_per_layer"] == 24
    assert r["working_set_mb"] == r["dense_equivalent_mb"]
    r = price_working_set({"lm": small}, {"lm": 100 * MB + small.routed_bytes}, 4096, 8192 * MB, "rung")
    assert r["expert_cache_mb"] == 0 and r["cached_experts_per_layer"] == 0


def test_a_graph_that_disagrees_with_its_declaration_is_refused():
    graph = {"tensors": {f"param::block.0.ffn.expert.{i}.{p}.weight":
                         {"is_parameter": True, "shape": [4, 8], "dtype": "bfloat16",
                          "weight_name": f"block.0.ffn.expert.{i}.{p}.weight"}
                         for i in range(4) for p in ("ffn_gate", "up", "down")},
             "ops": {"m": {"op_type": "custom::moe_fused", "attributes": {
                 "num_experts": 4, "top_k": 2, "norm_topk_prob": True,
                 **{f"expert_{r}_weight_ids": [f"param::block.0.ffn.expert.{i}.{p}.weight" for i in range(4)]
                    for r, p in (("gate", "ffn_gate"), ("up", "up"), ("down", "down"))}}}},
             "execution_order": ["m"]}
    t = read_expert_topology("lm", graph, None, 1.0, {"num_experts": 4, "num_experts_per_tok": 2}, "lm")
    assert t.top_k == [2] and t.layers[0].expert_bytes == 3 * 4 * 8 * 2
    with pytest.raises(RuntimeError, match="num_experts_per_tok=3"):
        read_expert_topology("lm", graph, None, 1.0, {"num_experts": 4, "num_experts_per_tok": 3}, "lm")
    # another component's declaration is not this one's
    assert read_expert_topology("other", graph, None, 1.0, {"num_experts_per_tok": 3}, "lm").top_k == [2]


def test_a_plan_states_the_working_set_in_its_record():
    from neurobrix.core.prism.solver import ExecutionPlan, explain_plan, plan_record
    t = ExpertTopology("lm", [_layer(128, 8, 9 * MB) for _ in range(2)], shared_experts=0)
    ews = price_working_set({"lm": t}, {"lm": t.routed_bytes + MB}, 32768, 0, "a rung")
    plan = ExecutionPlan(components={}, target_dtype="float16", total_memory_mb=0.0, strategy="zero3", component_memory={},
                         expert_working_set=ews)
    assert plan_record(plan)["expert_working_set"]["working_set_mb"] == ews["working_set_mb"]
    assert "stated, not used to place" in explain_plan(plan)
    plan.expert_working_set = None
    assert "expert_working_set" not in plan_record(plan)
