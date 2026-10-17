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

from neurobrix.core.prism.solver import ComponentMemory, PrismSolver
from tests.conftest import installed_container

MB = 1024 * 1024
MODEL = "CogVideoX-2b"


class _Container:
    def __init__(self, path):
        self._cache_path = Path(path)


def _spec(monkeypatch, lattice):
    root = installed_container(MODEL)
    monkeypatch.setattr(PrismSolver, "_tile_extent_lattice", staticmethod(lambda: lattice))
    s = PrismSolver()
    s._mode = "compiled"
    # the stub container names no components: the dtype a real container declares for the VAE
    monkeypatch.setattr(s, "_get_component_dtype", lambda c, n: "float32", raising=False)
    monkeypatch.setattr(s, "_run_dtype", lambda n: "float32", raising=False)   # the dtype the cut runs at
    monkeypatch.setattr(s, "_activation_widths", lambda *a, **k: None, raising=False)   # the stub has no profile
    from neurobrix.core.prism.profiler import InputConfig
    s._input_config = InputConfig(batch_size=2, height=352, width=720, num_frames=49, vae_scale=8,
                                  temporal_compression=4)
    mem = ComponentMemory("vae", 471 * MB, 121275 * MB, 6087 * MB)       # the live plan's own figures
    return s._spatial_component_tiling(_Container(root), "vae", mem, 12288)


def test_the_live_case_tiles_on_the_card(monkeypatch):
    """The live case stays on the card — since 2026-10-17 in its VENDOR's tiles (CogVideoX's declared tiled_decode,
    each tile in the vendor's frame batches): its group norm makes an engine tile another function."""
    spec = _spec(monkeypatch, 16)
    assert spec is not None, "a tile that fits was traded for host execution"
    assert spec.get("vendor") == "ramp_crop" and spec.get("tile_stream"), spec
    assert spec["tiled_activation_bytes"] <= 12288 * MB, spec


def test_the_vendor_cut_does_not_follow_the_engine_lattice(monkeypatch):
    """The lattice is the ENGINE tile's alignment; the vendor's tiles are the vendor's numbers, at every lattice."""
    specs = [_spec(monkeypatch, lattice) for lattice in (0, 8, 16)]
    assert all(sp is not None and sp.get("vendor") == "ramp_crop" for sp in specs), specs
    assert len({(sp["tiled_activation_bytes"], tuple(sp["tile_stream"])) for sp in specs}) == 1, specs
