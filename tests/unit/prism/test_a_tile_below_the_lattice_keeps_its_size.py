"""A tile the budget sizes below one lattice unit keeps its size; the lattice never pushes it over the budget.

2026-09-27, the regression matrix: CogVideoX-2b and CogVideoX-5b-I2V at 352x720x49 on a 16 GB V100 planned
'vae -> cpu' and decoded 45 minutes on the host. Live, _spatial_component_tiling sized a 12-latent tile under the
12 288 MB rung (budget 4.80 GB) and then snapped it onto Volta's extent lattice with max(16, (12 // 16) * 16) = 16 —
UP, over the budget — and returned None ("spatial-only insufficient"); offline, where no vendor profile is read, the
same request tiled on the card. The lattice is a performance alignment the comment says is taken DOWN; below one unit
there is nothing to take down to, and the sized tile is kept. On the old code the lattice-16 case returns None.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from neurobrix.core.paths import cache_dir
from neurobrix.core.prism.solver import ComponentMemory, PrismSolver

MB = 1024 * 1024
CACHE = cache_dir()
MODEL = "CogVideoX-2b"


class _Container:
    def __init__(self, path):
        self._cache_path = Path(path)


def _spec(monkeypatch, lattice):
    if not (CACHE / MODEL).exists():
        pytest.skip(f"{MODEL} is not in this machine's cache")
    monkeypatch.setattr(PrismSolver, "_tile_extent_lattice", staticmethod(lambda: lattice))
    s = PrismSolver.__new__(PrismSolver)
    from neurobrix.core.prism.profiler import InputConfig
    s._input_config = InputConfig(batch_size=2, height=352, width=720, num_frames=49, vae_scale=8)
    mem = ComponentMemory("vae", 471 * MB, 121275 * MB, 6087 * MB)       # the live plan's own figures
    return s._spatial_component_tiling(_Container(CACHE / MODEL), "vae", mem, 12288)


def test_the_live_case_tiles_on_the_card(monkeypatch):
    spec = _spec(monkeypatch, 16)
    assert spec is not None, "a tile that fits was traded for host execution"
    assert spec["tile_size"] == 12 and spec["tiled_activation_bytes"] <= int(12288 * 0.40 * MB), spec


def test_a_tile_at_or_above_the_unit_is_still_snapped_down(monkeypatch):
    big = _spec(monkeypatch, 8)          # 12 on a lattice of 8 -> 8, never more
    assert big is not None and big["tile_size"] == 8, big


def test_no_lattice_is_the_sized_tile(monkeypatch):
    assert _spec(monkeypatch, 0)["tile_size"] == 12
