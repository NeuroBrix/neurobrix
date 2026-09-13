"""The consumed-weight filter is applied in the loader's key space.

2026-09-13: Wan2.2's text encoder rendered in compiled mode and died in triton
at aten.embedding::0 with a None weight. The graph names the embedding
`encoder.token_embed.weight`; the index stores `token_embed.weight`; the
`only=` filter compared the two by exact membership BEFORE the suffix
reconciliation that would have joined them, so the weight was never loaded.

Run: PYTHONPATH=src python -m pytest tests/unit/runtime/test_consumed_weights_reach_the_loader_key_space.py
"""
from neurobrix.core.runtime.graph_executor import GraphExecutor


def test_a_prefix_difference_does_not_drop_the_weight():
    consumed = {"encoder.token_embed.weight", "encoder.block.0.attn.key.weight"}
    index = ["token_embed.weight", "encoder.block.0.attn.key.weight", "lm_head.weight"]
    wanted = GraphExecutor.consumed_in_loader_space(consumed, index)
    assert wanted == {"token_embed.weight", "encoder.block.0.attn.key.weight"}, wanted
    assert "lm_head.weight" not in wanted, "a weight no op consumes stays out — the saving the filter exists for"


def test_none_means_everything_as_before():
    assert GraphExecutor.consumed_in_loader_space(None, ["a", "b"]) is None
