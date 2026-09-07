"""The certifier's float64 oracle on windows of a large convolution's output."""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels import autotune_certify as Z


@pytest.mark.parametrize("groups, stride, padding, dilation", [(1, (1, 1), (1, 1), (1, 1)), (1, (2, 2), (1, 1), (1, 1)),
                                                              (8, (1, 1), (2, 2), (2, 2)), (4, (2, 1), (0, 1), (1, 1))])
def test_a_window_of_the_oracle_is_the_full_oracles_block(groups, stride, padding, dilation):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 8, 17, 19)); w = rng.standard_normal((16, 8 // groups, 3, 3))
    full = Z._conv2d_oracle(x, w, stride, padding, dilation, groups)
    n, co, oh, ow = full.shape
    for win in ((0, 0, 4, 0, 5), (1, 3, oh, 2, ow), (2, oh - 2, oh, ow - 3, ow)):
        ni, r0, r1, c0, c1 = win
        part = Z._conv2d_oracle(x, w, stride, padding, dilation, groups, window=win)
        assert part.shape == (1, co, r1 - r0, c1 - c0)
        assert np.allclose(part[0], full[ni, :, r0:r1, c0:c1], rtol=0, atol=1e-12)


def test_the_depthwise_oracle_is_unchanged_by_the_blas_form():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((1, 6, 9, 9)); w = rng.standard_normal((6, 1, 3, 3))
    out = Z._conv2d_oracle(x, w, (1, 1), (1, 1), (1, 1), 6)
    ref = np.zeros_like(out)
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    for c in range(6):
        for i in range(9):
            for j in range(9):
                ref[0, c, i, j] = (xp[0, c, i:i + 3, j:j + 3] * w[c, 0]).sum()
    assert np.allclose(out, ref, atol=1e-12)


def test_small_shapes_take_the_whole_oracle_and_large_ones_three_corners_and_the_centre():
    assert Z._conv_windows(1, 32, 32, 64, 64, 3, 3) is None
    wins = Z._conv_windows(49, 1024, 1024, 512, 512, 3, 3)
    assert len(wins) == 3
    assert wins[0][0] == 0 and wins[0][1] == 0 and wins[0][3] == 0                       # top-left, first frame
    assert wins[-1][0] == 48 and wins[-1][2] == 1024 and wins[-1][4] == 1024               # bottom-right, last frame
    per_position = 512 * 512 * 9
    for ni, r0, r1, c0, c1 in wins:
        assert (r1 - r0) * (c1 - c0) * per_position <= Z.ORACLE_MAX_MACS / 3 + per_position


def test_a_windowed_oracle_measures_its_windows_and_the_proof_names_them():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 4, 12, 12)); w = rng.standard_normal((4, 4, 3, 3))
    full = Z._conv2d_oracle(x, w, (1, 1), (1, 1), (1, 1), 1)
    wins = [(0, 0, 3, 0, 3), (1, 9, 12, 9, 12)]
    wo = Z.WindowedOracle([(win, Z._conv2d_oracle(x, w, (1, 1), (1, 1), (1, 1), 1, window=win)) for win in wins], 12, 12, 2)
    assert Z.oracle_deviation(full, wo) == 0.0
    bad = full.copy(); bad[1, :, 11, 11] *= 3                                            # a fault in the last corner is seen
    assert Z.oracle_deviation(bad, wo) > 0.5
    assert "2 window(s)" in wo.describe and "batch 1 rows 9-12 cols 9-12" in wo.describe


def test_a_conv_oracle_fn_is_windowed_only_above_the_cap(monkeypatch):
    rng = np.random.default_rng(4)
    x = rng.standard_normal((1, 4, 8, 8)).astype(np.float32); w = rng.standard_normal((4, 4, 3, 3)).astype(np.float32)
    assert isinstance(Z._conv_oracle_fn(x, w, (1, 1), (1, 1), (1, 1), 1)(), np.ndarray)
    monkeypatch.setattr(Z, "ORACLE_MAX_MACS", 100)
    o = Z._conv_oracle_fn(x, w, (1, 1), (1, 1), (1, 1), 1)()
    assert isinstance(o, Z.WindowedOracle) and o.describe.startswith("on ")
