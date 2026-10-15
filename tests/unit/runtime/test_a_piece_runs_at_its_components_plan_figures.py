"""A streamed component's pieces run at the plan figures the factory gave the component.

The factory sets the attention scores budget and the drain cliff on the component's executor (`ComponentAllocation`
-> `executor._attention_scores_budget_bytes`, `_deferred_drain_cliff_bytes`); layer streaming builds one executor per
piece from it. Before this, a piece carried neither: Allegro 720x1280 streamed on the Mac reached its first attention
with no budget and the compiled engine refused it by name (2026-10-15 16:02, main 333bc9a7c).

What this test does if the code is wrong: it fails when a piece's executor does not carry the base's budget or cliff.
"""
from neurobrix.core.strategies.layer_streaming import LayerStreamingStrategy


class _Base:
    def __init__(self, **kw):
        self.kw = kw
        self.loaded = None

    def load_graph_from_dict(self, sub):
        self.loaded = sub


def test_a_piece_carries_its_components_budget_and_cliff():
    base = _Base(family="video", vendor="apple", arch="apple_silicon", device="mps:0", dtype="float16", mode="compiled")
    for k, v in base.kw.items():
        setattr(base, k, v)
    base._attention_scores_budget_bytes = 1906940313
    base._deferred_drain_cliff_bytes = 123456789
    piece = LayerStreamingStrategy._piece_executor(base, {"ops": {}}, "transformer", base)
    assert piece._attention_scores_budget_bytes == 1906940313
    assert piece._deferred_drain_cliff_bytes == 123456789
    assert piece.loaded == {"ops": {}}
