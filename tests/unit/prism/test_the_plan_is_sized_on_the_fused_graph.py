"""Prism sizes a declared MoE LM on the FUSED graph — the one the engines run —
never on the container's graph, and never mutates the container's graph.

Run: PYTHONPATH=src python -m pytest tests/unit/prism/test_the_plan_is_sized_on_the_fused_graph.py
"""
from types import SimpleNamespace

from neurobrix.core.prism import solver as S


def _solver(lm):
    sv = S.PrismSolver.__new__(S.PrismSolver)
    sv._read_lm_config = lambda container: lm
    return sv


def test_a_declared_moe_is_sized_on_a_fused_copy(monkeypatch):
    seen = {}
    def fake_fuse(dag, family, norm_topk_prob=True, declared=False):
        seen["declared"] = declared; seen["norm"] = norm_topk_prob
        dag["ops"]["moe_fused::block.0"] = {"op_type": "custom::moe_fused", "input_tensor_ids": []}
        return dag
    import neurobrix.core.runtime.graph.moe_fusion as M
    monkeypatch.setattr(M, "detect_and_fuse_moe", fake_fuse)
    # a routed component: it holds the router's topk
    graph = {"tensors": {}, "ops": {"aten.topk::0": {"op_type": "aten::topk"}}, "execution_order": []}
    comp = SimpleNamespace(graph=graph, weights_index={"tensors": {}}, name="model")
    out = _solver({"num_experts": 8, "norm_topk_prob": False})._graph_as_executed(comp, None)
    assert seen == {"declared": True, "norm": False}
    assert "moe_fused::block.0" in out.graph["ops"]
    assert set(graph["ops"]) == {"aten.topk::0"}, "the container's graph is never mutated by the solver"


def test_a_tower_without_a_router_is_never_copied():
    graph = {"tensors": {}, "ops": {"aten.mm::0": {"op_type": "aten::mm"}}, "execution_order": []}
    comp = SimpleNamespace(graph=graph, weights_index=None, name="vision")
    assert _solver({"num_experts": 8})._graph_as_executed(comp, None) is comp


def test_a_dense_or_already_fused_component_is_returned_as_is():
    graph = {"tensors": {}, "ops": {"x": {"op_type": "custom::moe_fused"}}, "execution_order": []}
    comp = SimpleNamespace(graph=graph, weights_index=None, name="model")
    assert _solver({"num_experts": 8})._graph_as_executed(comp, None) is comp
    assert _solver(None)._graph_as_executed(comp, None) is comp
