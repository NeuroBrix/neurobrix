"""The request-reshape rung was refused for the entire upscaler family.

The engine can already cut a component's spatial input, run the pieces and stitch
them: `_spatial_component_tiling` sizes it, `plan.component_tiling` carries it, and
`executor.py` builds a `TilingEngine` from it. Residency is then bounded by the
PIECE, which is the property the op-level rung cannot give -- that one bounds the
transient while the full output stays allocated for downstream consumers.

It never fired for an upscaler, for two reasons, and both asked for a number the
model does not have while the container already held the answer.

ONE -- the scale factor. `config.get("upscale")`, then a VAE block list, then
`if not scale_factor: return None`. Every upscaler in this machine's cache carries
an EMPTY config: real-esrgan x2/x4/x8, swin2SR-classical-sr-x4-64, hat-l-x4. And
every one states its factor exactly in its own shapes:

    real-esrgan-x8   in [1,3,112,80]  -> out [1,3,896,640]   8 and 8
    real-esrgan-x4   in [1,3,112,80]  -> out [1,3,448,320]   4 and 4
    real-esrgan-x2   in [1,3,112,144] -> out [1,3,224,288]   2 and 2

TWO -- the latent grid. `if not (vae_scale and _h and _w): raise
MissingRuntimeValue(...)`, telling the operator to declare a VAE scale for a model
with no VAE. `InputConfig`'s own docstring already said that is legitimate: "absent
is legitimate only for a dimension the model does not have: no VAE, no vae_scale."
An upscaler reads pixels and writes pixels; its tiles cover the request's own grid.

With both read from the authority that knows them, `real-esrgan-x8` at 1024x1024
sizes at tile 578, overlap 72, **5,220 MB of activation against 16,384 MB
untiled** -- under a 5,232 MB budget it was previously refused outright for.

SEEN RED: with the graph-derived scale removed, `test_the_scale_comes_from_the_graph`
and `test_the_esrgan_case_is_now_sized` fail; with the `vae_scale` guard restored to
`if not (vae_scale and _h and _w)`, `test_no_vae_means_the_requests_own_grid` and
`test_the_esrgan_case_is_now_sized` fail.

Run: PYTHONPATH=src python -m pytest tests/unit/prism/test_the_reshape_rung_reaches_an_upscaler.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurobrix.core.prism.profiler import InputConfig
from neurobrix.core.prism.solver import ComponentMemory, PrismSolver

MB = 1024 * 1024
CACHE = Path("/home/mlops/.neurobrix/cache")
MODEL = "real-esrgan-x8"


class _Container:
    def __init__(self, path):
        self._cache_path = path


def _solver(h=1024, w=1024, vae_scale=None):
    s = PrismSolver.__new__(PrismSolver)
    s._input_config = InputConfig(batch_size=1, height=h, width=w,
                                  dtype="float16", vae_scale=vae_scale)
    s._component_tiling = {}
    return s


def _size(model=MODEL, h=1024, w=1024, act_mb=16384, budget_mb=5232, vae_scale=None):
    path = CACHE / model
    if not (path / "components" / "model" / "graph.json").exists():
        pytest.skip(f"{model} is not in this machine's cache")
    mem = ComponentMemory("model", 32 * MB, act_mb * MB, 821 * MB)
    return _solver(h, w, vae_scale)._spatial_component_tiling(
        _Container(path), "model", mem, budget_mb * MB)


def test_the_esrgan_case_is_now_sized():
    """The case that has refused four times tonight."""
    spec = _size()
    assert spec is not None, (
        "the rung still refuses real-esrgan-x8 at 1024x1024 — the request goes to "
        "the host instead of being cut into pieces it fits in")
    assert spec["scale_factor"] == 8, spec
    assert spec["tile_size"] > 0 and spec["overlap"] > 0, spec
    assert spec["tiled_activation_bytes"] < 16384 * MB, (
        "tiling must bound the residency below the untiled activation, or it "
        "bought nothing")


def test_the_scale_comes_from_the_graph_when_the_config_is_silent():
    """And the config really is silent — checked, not assumed."""
    prof = json.load(open(CACHE / MODEL / "components" / "model" / "profile.json"))
    assert not (prof.get("config") or {}).get("upscale"), (
        "this container now declares an upscale, so this cell no longer tests "
        "the derivation it was written for")
    assert _size()["scale_factor"] == 8


@pytest.mark.parametrize("model,expected", [
    ("real-esrgan-x2", 2), ("real-esrgan-x4", 4), ("real-esrgan-x8", 8),
])
def test_every_upscaler_in_the_cache_is_reached(model, expected):
    """One model could pass by coincidence; the family cannot."""
    spec = _size(model=model)
    assert spec is not None and spec["scale_factor"] == expected, (model, spec)


def test_no_vae_means_the_requests_own_grid():
    """A model with no VAE must not be asked to declare a VAE scale.

    Pinned by driving the same component with vae_scale absent AND present: both
    must size, and the absent case must not raise.
    """
    from neurobrix.core.runtime_values import MissingRuntimeValue
    try:
        spec = _size(vae_scale=None)
    except MissingRuntimeValue as e:                      # pragma: no cover
        pytest.fail(f"an upscaler was asked for a VAE scale it cannot have: {e}")
    assert spec is not None
    # a real latent model still uses its latent grid, so the branch is not a
    # blanket removal of the divisor
    tiled_at_8 = _size(vae_scale=8)
    assert tiled_at_8 is not None
    assert tiled_at_8["tile_size"] <= spec["tile_size"], (
        "with a vae_scale the grid is smaller, so the tile must not be larger")


def test_a_request_without_extents_still_refuses():
    """The guard that remains is the one that should: no height, no decision."""
    from neurobrix.core.runtime_values import MissingRuntimeValue
    s = _solver(h=None, w=None)
    mem = ComponentMemory("model", 32 * MB, 16384 * MB, 821 * MB)
    with pytest.raises(MissingRuntimeValue):
        s._spatial_component_tiling(_Container(CACHE / MODEL), "model", mem, 5232 * MB)


def test_a_component_that_fits_is_left_alone():
    """Tiling a component that already fits would be motion, not a remedy."""
    assert _size(act_mb=1024, budget_mb=5232) is None
