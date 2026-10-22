"""A softmax key carries its row length's CLASS (the padded row): it is certified at the class's top when the card
holds it in and out within the capacity Prism plans against, else at the longest row of the class that does — a
length that forms the same key — and refused by name when none does. HAT-L at 2048 px on 16 GB forms 2 097 152 rows
of 576 (class 1 024): drawn at 1 024 the key asked the class's whole 16 GiB and its certification failed at malloc
(2026-10-22).

Injection (seen red, then restored green, 2026-10-22): `_softmax_shape` returning the key unchanged -> red."""
from __future__ import annotations

import pytest

from neurobrix.kernels import autotune_certify as A
from neurobrix.kernels.launch_keys import softmax_feat_bucket

CARD = 16 * 1024 ** 3


def test_a_key_the_card_holds_is_drawn_at_its_class_top():
    assert A._softmax_shape(1024, 256, CARD, "fp32", "fp32", "softmax") == (1024, 256)


def test_a_key_over_the_card_is_drawn_at_the_shortest_row_of_its_class():
    rows, feat = A._softmax_shape(2097152, 1024, CARD, "fp32", "fp32", "softmax")
    assert rows == 2097152 and feat == 513 and softmax_feat_bucket(feat) == 1024, (rows, feat)


def test_a_key_no_row_of_its_class_fits_is_refused_by_name():
    with pytest.raises(A.KeyTooLargeForClass):
        A._softmax_shape(25165824, 1024, CARD, "fp32", "fp32", "softmax")
