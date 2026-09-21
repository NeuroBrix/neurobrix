"""apple_m4_pro declares a measured tile-extent lattice, and the solver reads it.

Before this the Apple profile carried no `tiling.extent_lattice`, so
`_tile_extent_lattice()` returned 0 and a tiled spatial extent reached the
conv kernels unaligned — running masked at every scale, ~3x slower (measured
synthetic conv 64->64 3x3 bf16, fixed config: odd 457 -> 20.9 ms, aligned
448 -> 7.9 ms; and 201 -> 4.6 ms, 208 -> 2.08 ms). The unit is 16, measured
on Metal the way volta.yml measured it for the V100 — Metal's 32-wide SIMD
notwithstanding, ÷32 buys nothing per unit area over ÷16.

The value is data in the profile, read through the one door the solver uses.
"""
from __future__ import annotations

import pytest


def test_the_profile_declares_the_measured_unit():
    from neurobrix.kernels.ops._configs import active_vendor_profile
    prof = active_vendor_profile()
    if not prof or "apple" not in str(prof.get("architecture", "")).lower():
        pytest.skip("this cell reads the ACTIVE Apple profile; not on Apple")
    lattice = (prof.get("tiling") or {}).get("extent_lattice")
    assert lattice == 16, (
        f"apple_m4_pro.tiling.extent_lattice is {lattice!r}, not the measured "
        f"16 — a change here moves how every tiled upscaler aligns; re-measure "
        f"before editing (the two-band sweep is in the profile comment)")


def test_the_yaml_carries_the_value_off_apple_too():
    """The file is data, checkable on any machine — the CI that is not a Mac
    must still catch a lattice silently dropped from the profile."""
    import pathlib
    import yaml

    p = (pathlib.Path(__file__).resolve().parents[3]
         / "src/neurobrix/config/vendors/apple/apple_m4_pro.yml")
    doc = yaml.safe_load(p.read_text())
    assert (doc.get("tiling") or {}).get("extent_lattice") == 16
