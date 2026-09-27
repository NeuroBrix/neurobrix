"""Tiles are crossfaded across their overlap, never box-averaged.

The component TilingEngine accumulated every tile with weight 1 and divided: the average
switches abruptly wherever the number of covering tiles changes, and a tile's edge (its conv
padding) weighs as much as its neighbour's interior. The regression matrix showed the result in
both engines — a line at every tile stride: CogVideoX-2b's VAE (tile 12, overlap 4, scale 8) a
64-px grid, Wan2.1-T2V's (tile 16, overlap 4) a 96-px one (2026-09-27). The vendors' tiled
decode crossfades the overlap (diffusers blend_h / blend_v); so does this engine now.
"""
from __future__ import annotations

import torch

from neurobrix.core.module.tiling_engine import TilingEngine


def _run(h, w, video=False):
    eng = TilingEngine(trace_size=8, overlap=4, scale_factor=2, tile_size=8)
    x = torch.zeros((1, 1, 3, h, w) if video else (1, 1, h, w))
    calls = []

    def execute(tile):
        # Each tile answers a constant of its own: a blend that is continuous across tiles
        # can only come from the weights.
        calls.append(len(calls))
        shape = list(tile.shape)
        shape[-2] *= 2
        shape[-1] *= 2
        return torch.full(shape, float(len(calls)) * 10.0)

    out = eng.tiled_execute(x, execute)
    return out, len(calls)


def test_the_blend_has_no_step_larger_than_one_ramp_step():
    out, n = _run(8, 20)                                   # one row of 4 tiles, stride 4
    assert n >= 3
    row = out[0, 0, 0]
    jumps = (row[1:] - row[:-1]).abs()
    # overlap 4 in input = 8 output pixels; a 10-unit difference crossfades in steps of 10/9
    assert float(jumps.max()) <= 10.0 / 8 + 1e-4, row.tolist()


def test_a_tile_interior_keeps_its_own_value():
    out, _ = _run(8, 20)
    assert float(out[0, 0, 0, 0]) == 10.0                  # the canvas border: first tile only


def test_a_video_tile_is_feathered_on_its_spatial_axes():
    out, _ = _run(8, 20, video=True)
    row = out[0, 0, 1, 0]
    assert float((row[1:] - row[:-1]).abs().max()) <= 10.0 / 8 + 1e-4
