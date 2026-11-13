"""A zero3 component's resident grant covers EVERY weight of the component, the non-block ones first.

The Triton loader keeps a zero3 component's non-block weights (embeddings, norms) on the card, so their
host bytes are zero; the grant then went to the blocks alone and the non-block weights sat beside it,
priced nowhere. Qwen2.5-Coder-14B --triton on one V100-16GB held 9 blocks on top of them and ran out of
memory at the second pass, short by 38 MB (2026-11-12 23:08)."""
from neurobrix.core.strategies.zero3 import Zero3Strategy


class _W:
    """A weight as the ratchet reads it: where it lives and its bytes."""

    def __init__(self, device: str, nbytes: int):
        self._device = device
        self._nbytes = nbytes


def _layout(nonblock_device: str):
    originals = {"nb": _W(nonblock_device, 1000)}
    blocks = {}
    for b in range(5):
        originals[f"b{b}"] = _W("cpu", 400)
        blocks[b] = {"first_op": b, "last_op": b, "weight_tensor_ids": [f"b{b}"]}
    return {"first_op": 0, "last_op": 0, "weight_tensor_ids": ["nb"]}, blocks, originals


def test_non_block_weights_on_the_card_are_charged_before_the_blocks():
    nonblock, blocks, originals = _layout("cuda:0")
    resident, spent = Zero3Strategy._resident_blocks(nonblock, blocks, originals, budget=2000)
    assert resident == {0, 1}          # 1000 already on the card + 2 x 400; a third block would pass 2000
    assert spent == 1800


def test_non_block_weights_on_the_host_are_made_resident_first():
    nonblock, blocks, originals = _layout("cpu")
    resident, spent = Zero3Strategy._resident_blocks(nonblock, blocks, originals, budget=2000)
    assert resident == {-1, 0, 1}
    assert spent == 1800


def test_the_card_count_reads_only_weights_already_on_a_card():
    nonblock, _, originals = _layout("cuda:0")
    assert Zero3Strategy._host_bytes(nonblock, originals) == 0
    assert Zero3Strategy._host_bytes(nonblock, originals, on_card=True) == 1000
