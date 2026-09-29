"""Under the Triton engine a host rung is planned on the card that computes it.

A component placed on `cpu` keeps its WEIGHTS on the host under both engines; the compiled engine
computes it there, the Triton engine computes it on the card (the weight loader's zero3 convention
for an all-`cpu` shard map). The plan kept a component's tiling only when it was placed on a card,
so a Triton `cpu_streaming` plan ran every spatial component untiled on the card, and said it
"WILL run": Wan2.1-VACE at 720x1280 on a 16 GB card died on its encoder's first convolution, a
57 GB allocation on cuda:0 (2026-09-29).

Injections, each seen RED: the rungs' device sizing removed -> the tiling and naming cases; the host
tilings not carried into the plan -> the tiling case; the reason line back to "WILL run" -> the
reason case.
"""
from __future__ import annotations

from neurobrix.core.prism.profiler import InputConfig
from neurobrix.core.prism.solver import PrismSolver
from neurobrix.nbx.container import NBXContainer
from tests.unit.prism._pinned_machine import (V100_16GB, container_root, impose_rung,
                                              pin_dedicated_card, profile)

MODEL = "Allegro"   # its confirmation request (256x640, 88 frames): the decoder overflows a 16 GB card


def _plan(monkeypatch, mode):
    """The host rung, forced (the engine's deterministic single-rung selection): the question is
    what that rung plans, not whether it wins."""
    pin_dedicated_card(monkeypatch, 16151, 267, "the rack's card 0, 2026-09-29")
    impose_rung(monkeypatch, 16384)
    monkeypatch.setenv("NBX_FORCE_STRATEGY", "cpu_streaming")
    c = NBXContainer.load(str(container_root(MODEL)))
    request = InputConfig(batch_size=2, height=256, width=640, num_frames=88, temporal_compression=4)
    return PrismSolver().solve_smart(c, profile(V100_16GB), request, mode=mode)


def test_the_triton_host_rung_tiles_what_the_card_computes(monkeypatch):
    plan = _plan(monkeypatch, "triton")
    assert plan.strategy == "cpu_streaming", plan.strategy
    tiling = getattr(plan, "component_tiling", None) or {}
    assert "vae" in tiling, f"the decoder runs whole on the card again: {tiling}"


def test_the_triton_host_rung_says_where_it_computes(monkeypatch):
    why = _plan(monkeypatch, "triton").selection_reason
    assert "WILL run" not in why, why
    assert "computes them" in why, why


def test_the_compiled_host_rung_computes_on_the_host_untiled(monkeypatch):
    plan = _plan(monkeypatch, "compiled")
    assert plan.strategy == "cpu_streaming", plan.strategy
    assert not (getattr(plan, "component_tiling", None) or {}), plan.component_tiling


def test_a_component_no_tiling_fits_is_named(monkeypatch):
    """The sizing itself, on a stub: a component whose live activations exceed the card and that
    no tiling fits is recorded with the card and its figure — the plan's reason names it."""
    from types import SimpleNamespace
    s = PrismSolver.__new__(PrismSolver)
    s._mode, s._host_device_tilings, s._host_device_overflow = "triton", {}, {}
    card = SimpleNamespace(device_string="cuda:0", capacity_mb=16384, tile_rung_mb=16384, free_mb=16000)
    s._usable_mb = lambda d: 14_000.0
    s._live_activation_mb = lambda m: m
    s._spatial_component_tiling = lambda c, n, m, r: (
        {"tiled_activation_bytes": 4_000 * 2 ** 20} if n == "vae" else None)
    s._size_host_placement_on_device("cpu_streaming", [("vae", 60_000.0), ("transformer", 32_000.0),
                                                       ("text_encoder", 200.0)], [card], None)
    assert set(s._host_device_tilings["cpu_streaming"]) == {"vae"}
    assert s._host_device_overflow["cpu_streaming"] == ("cuda:0", 14_000.0, {"transformer": 32_000.0})
    s._mode = "compiled"
    s._size_host_placement_on_device("cpu_streaming", [("vae", 60_000.0)], [card], None)
    assert "cpu_streaming" not in s._host_device_tilings, "the compiled engine computes on the host"
