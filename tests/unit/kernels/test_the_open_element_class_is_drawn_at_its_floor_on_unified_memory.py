"""An element-wise class is certified at its top where the card holds the copy, else at its floor (`_ew_count`). The
open class's key is its LABEL (68 719 476 736 elements on the four-row ladder), no count a run reaches. On a
unified-memory device the bound is the key's phase price (`_refuse_unified`), which priced the label: 2 TB of a 19 GB
card, so every Apple profile refused the open class and canary / granite-speech missed it under certified-only
(2026-10-24 13:43). What this test would do if the code were wrong: `_unified_padded_row` leaving the ew key at its
label -> the second case is refused, red (seen red before the fix, 2026-10-24)."""
from __future__ import annotations

import pytest

from neurobrix.kernels import autotune_certify as A
from nbxtensor.ops.autotune_bucket import class_floor

EW = "nbxtensor.ops.elementwise_launch.ew_launch_kernel"
UNIFIED = 19 * 1024 ** 3
LADDER = [(65536, 65536), (1048576, 1048576), (16777216, 16777216), (None, 68719476736)]


@pytest.fixture
def profile(monkeypatch):
    from nbxtensor.ops import autotune_bucket as B
    monkeypatch.setattr(A.C, "output_dtype", lambda tuner, key: "fp32")
    real = B.ladder_for
    monkeypatch.setattr(B, "ladder_for", lambda dim, profile=None: LADDER if dim == "elements" else real(dim, profile))


def test_a_class_whose_top_fits_is_drawn_at_its_top(profile):
    key = (16777216, "fp32", "fp32")
    assert A._unified_padded_row(EW, None, key, UNIFIED, "ew_launch_kernel") == key


def test_the_open_class_is_drawn_and_priced_at_its_floor(profile):
    key = (68719476736, "fp32", "fp32")
    drawn = A._unified_padded_row(EW, None, key, UNIFIED, "ew_launch_kernel")
    assert drawn == (class_floor("elements", 68719476736),) + key[1:] and drawn[0] == 16777217, drawn
    A._refuse_unified(EW, None, drawn, UNIFIED)
