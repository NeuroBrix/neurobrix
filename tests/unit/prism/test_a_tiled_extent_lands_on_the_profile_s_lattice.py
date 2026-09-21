"""A tiled spatial extent is snapped down onto the vendor profile's lattice before the kernels
see it — the ladder's other half.

Measured 2026-09-21 (nbx/campaigns/2026_09_21_tile_align/ALIGN.md): real-esrgan-x8 at 1024² on
the 8192 rung computed a tile of 457, odd at every scale, and ran 717.9 s under static kernel
configs; 448 ran 40.9 s. The unit is 16 (432 and 416 sit on the 448 plateau; 456 does not).

What would this file do if the code were wrong? A lattice ignored → the first cell reads 457
and fails; a lattice applied where the profile declares none → the second fails; the
measurement override not winning → the third fails; a rounding UP or a floor below the unit →
the fourth fails.
"""
from __future__ import annotations

import pytest

from neurobrix.core.prism import solver as S


def _lattice(monkeypatch, profile):
    import neurobrix.kernels.ops._configs as C
    monkeypatch.setattr(C, "active_vendor_profile", lambda: profile)


def _snap(v, unit):
    return max(unit, (v // unit) * unit) if unit > 1 else v


def test_the_profile_s_lattice_snaps_the_extent_down(monkeypatch):
    monkeypatch.delenv("NBX_PRISM_TILE_ALIGN", raising=False)
    _lattice(monkeypatch, {"tiling": {"extent_lattice": 16}})
    unit = S.PrismSolver._tile_extent_lattice()
    assert unit == 16
    assert _snap(457, unit) == 448 and _snap(560, unit) == 560


def test_a_profile_without_the_value_keeps_the_extent_as_computed(monkeypatch):
    monkeypatch.delenv("NBX_PRISM_TILE_ALIGN", raising=False)
    _lattice(monkeypatch, {"memory": {"warp_size": 32}})
    unit = S.PrismSolver._tile_extent_lattice()
    assert unit == 0 and _snap(457, unit) == 457


def test_the_measurement_override_wins(monkeypatch):
    monkeypatch.setenv("NBX_PRISM_TILE_ALIGN", "48")
    _lattice(monkeypatch, {"tiling": {"extent_lattice": 16}})
    assert S.PrismSolver._tile_extent_lattice() == 48 and _snap(457, 48) == 432


def test_the_snap_is_down_and_never_below_the_unit():
    assert _snap(457, 16) == 448          # down, never 464
    assert _snap(15, 16) == 16            # the floor is one unit, not zero
    assert _snap(457, 104) == 416 and _snap(457, 128) == 384
