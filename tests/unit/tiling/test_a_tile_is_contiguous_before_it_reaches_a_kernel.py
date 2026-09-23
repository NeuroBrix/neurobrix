"""Every tile the TilingEngine hands out is contiguous, interior ones included.

`_extract_tile` returns `input_tensor[sl]`. A slice STRIDES ACROSS THE PARENT: an interior
32-wide tile of a 64-wide image comes back with strides (16384, 4096, 64, 1) — row stride 64
for a 32-wide tile. Handed to a flat-indexed wrapper that assumes packed memory it resolves
to the wrong addresses and produces silent garbage, which is the rule this repository already
carries in `.claude/rules/kernels-and-triton.md`:

    A non-contiguous slice fed to a flat-indexed wrapper resolves to the wrong addresses and
    produces silent garbage — the Sana 4Kpx conv::55 incident, 94 % of elements past 1.0
    absolute. Audit every `x[:, :, ...]` and add `.contiguous()`; it short-circuits at zero
    cost when already contiguous.

THE ASYMMETRY IS THE SIGNATURE
------------------------------
An EDGE tile runs off the parent and is padded, and pad returns a fresh packed tensor — so
edge tiles were always contiguous and correct. Every INTERIOR tile was not. That does not
produce a uniformly broken image; it produces one whose tile INTERIORS are wrong and whose
seams are clean, which reads as a GRID.

Measured 2026-09-22, CogVideoX-2b on a 32 GB card, `[OpTiling] vae: dropped full-extent
op-level tiling (component-level tiling active)`: the artefact is a flat orange vertical
gradient with a visible grid of darker lines at roughly 8 by 6, in both the first and the
last frame. The run exited rc=0, wrote a valid 159 KB mp4 with correct dimensions and frame
count, and logged no error, no warning and no refusal.

This cell asserts the property, not the picture: a tile is contiguous when it leaves here.
Whether that is the WHOLE cause of the CogVideoX artefact is not claimed by this file — that
needs a re-run, and the re-run is the artefact's gate, not this one's.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neurobrix.core.module.tiling_engine import TilingEngine  # noqa: E402


def _engine(tile_size=32, t_tile=None):
    e = TilingEngine.__new__(TilingEngine)
    e.tile_size = tile_size
    e.t_tile = t_tile
    return e


@pytest.mark.parametrize("y,x,where", [(0, 0, "corner"), (16, 16, "interior"),
                                       (32, 0, "left edge"), (48, 48, "far corner")])
def test_every_4d_tile_leaves_contiguous(y, x, where):
    e = _engine()
    t = e._extract_tile(torch.randn(1, 4, 64, 64), y, x)
    assert t.is_contiguous(), f"the {where} tile is strided: {t.stride()}"
    assert tuple(t.shape[-2:]) == (32, 32)


@pytest.mark.parametrize("y,x", [(0, 0), (16, 16), (40, 40)])
def test_every_5d_tile_leaves_contiguous(y, x):
    """Video: [B, C, T, H, W]. The trailing two dims are still H, W."""
    e = _engine()
    t = e._extract_tile(torch.randn(1, 4, 6, 64, 64), y, x)
    assert t.is_contiguous(), f"5-D tile at ({y},{x}) is strided: {t.stride()}"


def test_the_temporal_slice_is_contiguous_too():
    e = _engine(t_tile=3)
    t = e._extract_tile(torch.randn(1, 4, 9, 64, 64), 16, 16, t=3)
    assert t.is_contiguous(), f"temporal tile is strided: {t.stride()}"
    assert t.shape[2] == 3


def test_an_interior_tile_was_the_broken_one_and_an_edge_tile_was_not():
    """The asymmetry, pinned as a fact about the INPUT rather than the output — so this
    cell keeps meaning something after the fix. It is what makes the artefact a grid."""
    x = torch.randn(1, 4, 64, 64)
    raw_interior = x[(slice(None), slice(None), slice(16, 48), slice(16, 48))]
    assert not raw_interior.is_contiguous(), (
        "an interior slice is contiguous on this torch — the premise of the defect is gone "
        "and this gate no longer describes anything")
    padded_edge = torch.nn.functional.pad(
        x[(slice(None), slice(None), slice(48, 64), slice(48, 64))],
        (0, 16, 0, 16), mode="replicate")
    assert padded_edge.is_contiguous(), "a padded edge tile is no longer contiguous"


def test_contiguity_is_free_when_already_contiguous():
    """The rule's own claim — it short-circuits at zero cost — as an identity check."""
    packed = torch.randn(1, 4, 32, 32)
    assert packed.is_contiguous()
    assert packed.contiguous().data_ptr() == packed.data_ptr(), (
        "contiguous() copied an already-packed tensor; the fix would then cost a copy per "
        "tile rather than nothing")
