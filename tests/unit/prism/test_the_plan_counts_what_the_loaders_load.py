"""Prism sizes a component on the set the loaders load: the weights the graph
consumes PLUS every non-block weight (embedding, head, norms — read by the
flows outside the graph). Counting the consumed set alone under-estimated,
the direction the estimator's own docstring calls unsafe (review, 2026-09-13).

Run: PYTHONPATH=src python -m pytest tests/unit/prism/test_the_plan_counts_what_the_loaders_load.py
"""
from types import SimpleNamespace

from neurobrix.core.prism.solver import _consumed_weight_bytes


def test_non_block_weights_are_budgeted_and_unrouted_experts_are_not():
    params = {"block.0.attn.key.weight": 100, "block.0.mlp.experts.7.down.weight": 1000,
              "token_embed.weight": 500, "norm.weight": 10}
    tensors = {f"param::{n}": {"is_parameter": True, "weight_name": n} for n in params}
    graph = {"tensors": tensors, "ops": {"op0": {"input_tensor_ids": ["param::block.0.attn.key.weight"]}},
             "execution_order": ["op0"]}
    index = {"tensors": {n: {"size_bytes": b} for n, b in params.items()}}
    comp = SimpleNamespace(graph=graph, weights_index=index)
    assert _consumed_weight_bytes(comp, 1.0) == 100 + 500 + 10
