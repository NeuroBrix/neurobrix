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


def test_the_price_door_prices_the_key_drawn_not_the_key_filed(profile, monkeypatch):
    # the re-proof under the published Metal wheels (the Mac's RUN 73, 2026-11-07): the door before any draw priced the
    # open class at its label — 2.2 TB of an 18 GB Mac — and refused it, though the key drawn is its floor
    seen = []

    class Stop(Exception):
        pass

    def door(qual, tuner, key, budget, floor):
        seen.append(key)
        raise Stop
    monkeypatch.setattr(A, "_refuse_unpriced", door)
    key = (68719476736, "fp32", "fp32")
    with pytest.raises(Stop):
        A.synthesize(EW, None, key, None, card_bytes=UNIFIED, unified=True, values=False, budget_bytes=UNIFIED,
                     floor_bytes=0)
    assert seen == [(16777217, "fp32", "fp32")], seen


def test_a_windowed_oracle_keeps_only_its_windows_rows_on_the_host():
    # the Mac's re-proof (2026-11-07): softmax 2 621 440 x 1 024 fp32 held its 10 GB host draw whole for an oracle that
    # reads three windows; the price counts the windows' rows, and the oracle computes the same numbers from them
    import numpy as np
    SM = "nbxtensor.ops.attention_wrappers.softmax_forward_kernel"
    key = (2621440, 1024, False, "fp32", "fp32")
    pr = A.price_key(SM, None, key)
    whole = 2621440 * 1024 * 4
    assert pr["draws"] < whole // 10, pr                 # windows, not the draw
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4096, 64)).astype(np.float32)
    wins = [(0, 7), (2000, 2007), (4089, 4096)]
    hw = A._HostWindows(x, wins)
    for r0, r1 in wins:
        assert np.array_equal(hw[r0:r1, :], x[r0:r1, :])
    assert hw.shape == x.shape
