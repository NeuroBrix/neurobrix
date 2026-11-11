"""A card's pointer table of a routed layer keeps the LAYER's length, each expert at its own number.

The grouped GEMM reads a table at the expert's number in the layer, and its launch's key carries the table's length
(`NUM_EXPERTS`). When a plan leaves one layer's experts on two cards, each card's table held its own experts alone,
packed: a read at another expert's entry, or past the table's end (Qwen3-Coder-30B-A3B on two V100-32GB, 2026-11-11),
and a key per split count that no certified store can hold.

What this test would do if the code were wrong: a packed table for experts {1, 3} of 4 has two entries — RED on the
length and on the entry read at number 3."""
import numpy as np

from nbxtensor.memory.pointer_tables import _at_layer_numbers


def test_a_split_layers_table_is_read_at_the_experts_number():
    table = _at_layer_numbers(4, [1, 3], [0x1000, 0x3000])
    assert table.dtype == np.int64 and table.tolist() == [0, 0x1000, 0, 0x3000]


def test_a_layer_held_by_one_card_keeps_the_table_it_had():
    addresses = [0x10 * (e + 1) for e in range(6)]
    assert _at_layer_numbers(6, list(range(6)), addresses).tolist() == addresses
