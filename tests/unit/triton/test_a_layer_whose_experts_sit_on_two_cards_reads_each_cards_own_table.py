"""A routed layer whose experts are held by two cards computes what it computes on one (CARDS REQUIRED: two).

A plan that fills the cards in order (`weight_sharding`) can leave one layer's experts on both. Three defects of the
split path (now the library's op, `nbxtensor.ops.moe_multi_device.fused_moe_across_devices`, whose own test is
`nbxtensor/tests/test_a_layer_on_several_devices_is_the_sum_of_their_shares.py`; this one enters through the engine's
`execute_moe_fused`, which hands a split layer to it), found on Qwen3-Coder-30B-A3B on two V100-32GB, 2026-11-11
(89 experts of a block on one card, 39 on the other: "cudaMemcpy rc=700" on a run, a wrong text served):

  * each card held a pointer table of ITS experts only, packed in ascending order, and the kernel reads a table at
    the expert's number in the LAYER — right only for a card holding experts 0..n-1, another expert's weights for
    any other card, a read past the table's end for numbers beyond the card's count. A card's table now keeps the
    layer's length (`nbxtensor.memory.pointer_tables._at_layer_numbers`), which also keeps the launch's key the
    layer's (the table's length is in it): a packed table keyed every split layer apart, each swept at run time;
  * the kernel writes a pair's row at its global flat position, and the three outputs were sized by the card's own
    padded count — rows written past the buffer when a card receives few of the layer's pairs;
  * the combine read row i as the pair of sorted id i, where row p is the pair at flat position p.

What this test would do if the code were wrong: with experts {0, 2} on the first card and {1, 3} on the second, a
token routed to expert 1 is computed with expert 3's weights — O(1) against the fp64 oracle (case `in_table`: 1.54
seen before the fix, 1.46 with the packed tables alone put back); a token routed to expert 2 or 3 reads past a two-entry
table (case `every_expert`, which is why the red was taken on `in_table`). `one_card` is the control of the oracle
itself: the same weights on one card."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("triton")

from nbxtensor.tensor import NBXTensor  # noqa: E402

E, K_, H, F, T = 4, 2, 64, 32, 6


def _oracle(h, probs, gate, up, down):
    h = h.astype(np.float64)
    idx = np.argsort(-probs, axis=-1, kind="stable")[:, :K_]
    sc = np.take_along_axis(probs.astype(np.float64), idx, -1)
    sc = sc / sc.sum(-1, keepdims=True)
    out = np.zeros((T, H))
    for t in range(T):
        for e, s in zip(idx[t], sc[t]):
            g = h[t] @ gate[e].astype(np.float64).T
            u = h[t] @ up[e].astype(np.float64).T
            out[t] += s * (((g / (1.0 + np.exp(-g))) * u) @ down[e].astype(np.float64).T)
    return out


@pytest.mark.parametrize("routed", ["one_card", "in_table", "every_expert"])
def test_each_card_reads_its_own_table(routed, needs_device, declared_profile):
    from nbxtensor.memory.allocator import DeviceAllocator
    from neurobrix.triton.moe import _xfer, execute_moe_fused
    if DeviceAllocator.device_count() < 2:
        pytest.skip("two cards are needed: one layer's experts on both")
    rng = np.random.default_rng(11)
    h = (rng.standard_normal((T, H)) * 0.5).astype(np.float16)
    logits = rng.standard_normal((T, E)).astype(np.float32)
    if routed == "in_table":
        logits[:, 2:] -= 30.0                          # every token routes to experts 0 and 1
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs = (probs / probs.sum(-1, keepdims=True)).astype(np.float32)
    gate = (rng.standard_normal((E, F, H)) * 0.1).astype(np.float16)
    up = (rng.standard_normal((E, F, H)) * 0.1).astype(np.float16)
    down = (rng.standard_normal((E, H, F)) * 0.1).astype(np.float16)

    hs, gs = NBXTensor.from_numpy(h), NBXTensor.from_numpy(probs)      # the first allocation: no card -> skipped
    here = hs._device_idx
    other = next(d for d in range(DeviceAllocator.device_count()) if d != here)

    def on_cards(w):
        return [NBXTensor.from_numpy(w[e]) if e % 2 == 0 or routed == "one_card" else _xfer(NBXTensor.from_numpy(w[e]), other)
                for e in range(E)]
    lists = [on_cards(w) for w in (gate, up, down)]
    DeviceAllocator.set_device(here)
    DeviceAllocator.ensure_triton_device(here)
    out = execute_moe_fused(gs, hs, *lists, top_k=K_, num_experts=E, norm_topk_prob=True,
                            cache_key=f"experts-on-two-cards-{routed}")
    got = np.asarray(out.numpy(), dtype=np.float64).reshape(T, H)
    ref = _oracle(h, probs, gate, up, down)
    rel = np.abs(got - ref).max() / np.abs(ref).max()
    assert rel < 2e-2, f"{routed}: max rel {rel:.3e} against the oracle"
