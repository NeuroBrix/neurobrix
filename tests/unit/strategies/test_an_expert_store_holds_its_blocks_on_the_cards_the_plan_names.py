"""S3, the engine half: the routed layers of the blocks Prism holds WHOLE on other cards (`expert_cards`) are copied
onto those cards and published as `_expert_cards` {layer key: (card, (gate, up, down))}, each layer's block read as
Prism reads it (`expert_topology._block_of`); a block the plan holds with no routed layer of the graph is refused by
name. Qwen3-30B-A3B on 4 V100-16GB (2026-10-28): 31 blocks held on cards 0, 1, 3, decode 449 ms per token.

What this test would do if the code were wrong: a held layer missing from `_expert_cards`, on the wrong card, or a
layer of another block published (red); a plan naming a block the graph does not route, accepted (red)."""
from types import SimpleNamespace

import pytest

from neurobrix.core.prism.expert_topology import _LAYERS
from neurobrix.core.strategies import expert_store as ES


class _Holder:
    made = []

    def __init__(self, device, layers):
        self.device, self.layers, self.bytes = device, {k: tuple(zip(*v)) for k, v in layers.items()}, 0
        _Holder.made.append(self)


def _setup(monkeypatch, cards):
    import neurobrix.triton.expert_store as TS
    monkeypatch.setattr(TS, "NBXHeldExperts", _Holder)
    _Holder.made = []
    layers, tensors = {}, {}
    for b in range(3):
        ids = tuple([f"param::{_LAYERS}.{b}.mlp.experts.{e}.{p}.weight" for e in range(2)] for p in ("g", "u", "d"))
        for x in ids:
            for t in x:
                tensors[t] = {"weight_name": t[len("param::"):]}
        layers[ids[0][0]] = ids
    z3 = SimpleNamespace(context=SimpleNamespace(allocations={"model": {"expert_cards": cards}}),
                         _snapshot_cpu_originals=lambda ex, blocks: {t: f"host:{t}" for t in blocks[0]["weight_tensor_ids"]})
    ex, seq = SimpleNamespace(_dag={"tensors": tensors}), SimpleNamespace()
    return z3, ex, seq, layers


def test_the_layers_of_held_blocks_are_published_on_their_card(monkeypatch):
    z3, ex, seq, layers = _setup(monkeypatch, {"cuda:1": {"blocks": [1, 2], "cache_mb": 1, "experts": 4}})
    holders = ES._hold_experts_on_cards(z3, "model", ex, seq, True, layers)
    keys = sorted(layers)
    held = seq._expert_cards
    assert ex._expert_cards is held and len(holders) == 1 and holders[0].device == 1
    assert set(held) == {k for k in keys if f".1." in k or f".2." in k}, sorted(held)
    assert all(card == "cuda:1" for card, _ in held.values())
    k1 = next(k for k in keys if ".1." in k)
    assert held[k1][1][0][0] == f"host:{layers[k1][0][0]}"      # gate of expert 0, from the host copy


def test_a_held_block_the_graph_does_not_route_is_refused(monkeypatch):
    z3, ex, seq, layers = _setup(monkeypatch, {"cuda:1": {"blocks": [2, 5], "cache_mb": 1, "experts": 8}})
    with pytest.raises(RuntimeError, match=r"blocks \[3, 4, 5\]"):
        ES._hold_experts_on_cards(z3, "model", ex, seq, True, layers)


def test_without_expert_cards_nothing_is_held(monkeypatch):
    z3, ex, seq, layers = _setup(monkeypatch, {})
    assert ES._hold_experts_on_cards(z3, "model", ex, seq, True, layers) == [] and seq._expert_cards is None
