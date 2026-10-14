"""A plan that is not streamed holds a TILED component at its tile when it prices the moment it holds most.

The Mac, 2026-10-14: the harness reserved 419-421 GB of host for each of the four Allegro cells on its 24 GB
unified pool. The reservation is the plan's own host figure (`host_footprint.total_bytes`), and on unified
memory that adds the device bytes the plan holds at its dearest phase (`unified_device_bytes`,
`PrismSolver._peak_loaded_bytes`). That phase priced every component at its UNTILED total: Allegro's VAE,
tiled by the same plan to 6.7 GB of activations per tile, counted 415 GB of activations. Under the Mac's profile
(read without executing) the host figure was 420 970 MB; it is 17 085 MB with the tile.

What this test does if the code is wrong: the tiled cell fails when a tiled component is priced at its untiled
total (red on main); the untiled cell fails when a component the plan does not tile loses its whole figure; the
device cell fails when a tiled component whose device the plan does not name is silently priced at anything but
its total.

Run: python -m pytest tests/unit/prism/test_a_tiled_component_is_held_at_its_tile.py
"""
from types import SimpleNamespace as NS

from neurobrix.core.prism.solver import PrismSolver

MIB = 1024 * 1024


def _solver():
    s = PrismSolver.__new__(PrismSolver)
    s._lm_component_name = None
    s._phase_peak = lambda container, totals, outliving=0, owner=None: dict(totals)   # every phase's holding, read back
    s._get_component_dtype = lambda container, name: "bfloat16"
    return s


def _plan(device="mps:0"):
    return NS(
        component_memory={"vae": NS(total_bytes=415 * 1024 * MIB, weight_mb=211.6),
                          "transformer": NS(total_bytes=12 * 1024 * MIB, weight_mb=5287.0)},
        components={"vae": NS(device=device, strategy="lazy"), "transformer": NS(device="mps:0", strategy="lazy")},
        component_tiling={"vae": {"tiled_activation_bytes": 6400 * MIB}},
        layer_stream_plan=None, kv_cache_plan=None)


DEV = NS(device_string="mps:0", get_cost_multiplier=lambda dtype: 1.0)


def test_a_tiled_component_is_held_at_its_tile():
    held = _solver()._peak_loaded_bytes(None, _plan(), [DEV])
    assert held["vae"] == int((211.6 + 6400) * MIB), held["vae"] / MIB


def test_a_component_the_plan_does_not_tile_keeps_its_whole_figure():
    held = _solver()._peak_loaded_bytes(None, _plan(), [DEV])
    assert held["transformer"] == 12 * 1024 * MIB


def test_a_tiled_component_on_a_device_the_list_does_not_name_keeps_its_total():
    held = _solver()._peak_loaded_bytes(None, _plan(device="mps:1"), [DEV])
    assert held["vae"] == 415 * 1024 * MIB
