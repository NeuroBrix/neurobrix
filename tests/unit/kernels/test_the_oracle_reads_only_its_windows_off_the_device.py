"""A windowed oracle refuses to compute the whole float64 reference — that is what stopped a
certifier holding 155 GB of host memory (2026-09-21 23:32). It was still handed the whole
RESULT: the windows were cut after the crossing, so every candidate configuration copied the
kernel's entire output to the host so that three small windows could be read out of it. On a
49x128x96x720 convolution that is 1.73 GB a candidate and eighteen candidates a key, and the
certifier's own record shows it: `runs` 37.9 s against `bench` 2.7 s on keys whose kernel
takes milliseconds.

The windows are now cut on the tensor where it lives. This test pins that the cut is the SAME
one — the comparison must not change, only where the bytes are read — and that a window cut
wrongly is loud rather than silent.

What this test would do if the code were wrong: a device cut at the wrong offset yields values
that do not match the reference, so the deviation leaves the tolerance and the last case
fails; a cut of the wrong SHAPE makes the comparison return inf and the first cases fail.

Shapes: the convolution window is [1, 8, 4, 5] out of a 3x8x16x20 output, taken at the last
batch element's bottom-right corner — the corner is where a wrong tiling shows and where an
off-by-one in the cut would be invisible if the window were taken at the origin. The matrix
windows are rows 0-2 and 5-7 of an 8x6 product, and the same rows of a batched 4x8x6 one,
because the row cut is the one that has to know its tensor's rank.
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels.autotune_certify import (RowWindowedOracle, WindowedOracle,
                                                deviation_against, oracle_deviation)


def _conv_case():
    rng = np.random.default_rng(23)
    out = rng.standard_normal((3, 8, 16, 20)).astype(np.float32)
    win = (2, 12, 16, 15, 20)                       # last batch element, bottom-right corner
    ni, r0, r1, c0, c1 = win
    ref = out[ni:ni + 1, :, r0:r1, c0:c1].astype(np.float64)
    return out, WindowedOracle([(win, ref)], 16, 20, 3)


@pytest.mark.parametrize("shape", [(8, 6), (4, 8, 6)])
def _row_case(shape):
    pass


def _rows(shape):
    rng = np.random.default_rng(29)
    out = rng.standard_normal(shape).astype(np.float32)
    blocks = []
    for r0, r1 in ((0, 2), (5, 7)):
        blocks.append(((r0, r1), out[..., r0:r1, :].astype(np.float64)))
    return out, RowWindowedOracle(blocks, shape[-2])


def test_the_convolution_window_cut_on_the_tensor_is_the_window_cut_on_the_host():
    out, oracle = _conv_case()
    host = [np.asarray(p) for p, _ in oracle.slices(out)]
    dev = [np.asarray(p) for p, _ in oracle.device_slices(out)]
    assert len(host) == len(dev) == 1
    assert host[0].shape == dev[0].shape, (host[0].shape, dev[0].shape)
    assert np.array_equal(host[0], dev[0])


@pytest.mark.parametrize("shape", [(8, 6), (4, 8, 6)])
def test_the_row_windows_are_cut_the_same_at_every_rank(shape):
    out, oracle = _rows(shape)
    host = [np.asarray(p) for p, _ in oracle.slices(out)]
    dev = [np.asarray(p) for p, _ in oracle.device_slices(out)]
    assert [h.shape for h in host] == [d.shape for d in dev], (host, dev)
    for h, d in zip(host, dev):
        assert np.array_equal(h, d)


@pytest.mark.parametrize("case", ["conv", "rows2", "rows3"])
def test_the_deviation_is_the_one_the_whole_crossing_measured(case):
    out, oracle = _conv_case() if case == "conv" else _rows((8, 6) if case == "rows2" else (4, 8, 6))
    assert deviation_against(out, oracle) == oracle_deviation(out, oracle) == 0.0


def test_a_window_cut_at_the_wrong_place_is_loud():
    """The safety of reading less: a wrong cut cannot pass quietly, because its values are
    not the reference's."""
    out, oracle = _conv_case()
    (ni, r0, r1, c0, c1), ref = oracle.blocks[0]
    oracle.blocks = [((ni, r0 - 1, r1 - 1, c0, c1), ref)]        # one row off
    assert deviation_against(out, oracle) > 0.1, deviation_against(out, oracle)
