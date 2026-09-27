"""A tiled VAE decode is sized at the scale its own graph measures, not at a block-count heuristic.

SANA-Video_2B_720p_diffusers' VAE declares decoder_block_out_channels [256, 512, 1024]; Prism took
2^(3-1) = 4 for its spatial factor, while the graph maps a 14x22 latent to 448x704 pixels — 32.
Wherever the VAE tiled (lazy_sequential on a 32 GB V100, 2026-09-27) the tiled decode stitched a
canvas at latent x 4: a 160x64 video for a 1280x512 request, in the compiled engine and in
triton-sequential alike. Over the whole cache the graph and the heuristic disagree on this
container only; everywhere else the factor is unchanged.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from neurobrix.core.paths import cache_dir
from neurobrix.core.prism.solver import ComponentMemory, PrismSolver

MB = 1024 * 1024
CACHE = cache_dir()


class _Container:
    def __init__(self, path):
        self._cache_path = Path(path)


def _spec(monkeypatch, model, h, w, frames, weights_mb, act_mb, over_mb, rung_mb):
    if not (CACHE / model).exists():
        pytest.skip(f"{model} is not in this machine's cache")
    # The factor does not depend on the extent lattice; none, so the lattice's own rules stay out.
    monkeypatch.setattr(PrismSolver, "_tile_extent_lattice", staticmethod(lambda: 0))
    s = PrismSolver.__new__(PrismSolver)
    from neurobrix.core.prism.profiler import InputConfig
    s._input_config = InputConfig(batch_size=2, height=h, width=w, num_frames=frames, vae_scale=32)
    mem = ComponentMemory("vae", weights_mb * MB, act_mb * MB, over_mb * MB)
    return s._spatial_component_tiling(_Container(CACHE / model), "vae", mem, rung_mb)


def test_sana_video_tiles_at_the_scale_its_graph_measures(monkeypatch):
    spec = _spec(monkeypatch, "SANA-Video_2B_720p_diffusers", 512, 1280, 81, 2016, 20000, 10501, 32768)   # over the 32 GB rung budget: it tiles
    assert spec is not None
    assert spec["scale_factor"] == 32, spec


def test_a_vae_whose_heuristic_agrees_keeps_its_factor(monkeypatch):
    spec = _spec(monkeypatch, "CogVideoX-2b", 352, 720, 49, 471, 121275, 6087, 12288)
    assert spec is None or spec["scale_factor"] == 8, spec
