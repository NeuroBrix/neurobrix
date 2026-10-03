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
    assert "computes each component" in why, why


def test_the_compiled_host_rung_computes_on_the_host_untiled(monkeypatch):
    plan = _plan(monkeypatch, "compiled")
    assert plan.strategy == "cpu_streaming", plan.strategy
    assert not (getattr(plan, "component_tiling", None) or {}), plan.component_tiling


def _stub(mode="triton"):
    from types import SimpleNamespace
    s = PrismSolver.__new__(PrismSolver)      # never solved: the sizing must not need solve's state
    s._mode = mode
    card = SimpleNamespace(device_string="cuda:0", capacity_mb=16384, tile_rung_mb=16384, free_mb=16000)
    s._usable_mb = lambda d: 14_000.0
    s._live_activation_mb = lambda m: m.activation_mb
    s._whole_component_mb = lambda c, n, m, d: m.weight_mb + m.activation_mb
    s._spatial_component_tiling = lambda c, n, m, r: (
        {"tiled_activation_bytes": 4_000 * 2 ** 20} if n == "vae" else None)
    s._tiled_component_mb = lambda c, n, m, d, t: m.weight_mb + t["tiled_activation_bytes"] / 2 ** 20
    return s, card


def _m(w, a):
    from types import SimpleNamespace
    return SimpleNamespace(weight_mb=w, activation_mb=a, total_mb=w + a, weight_bytes=int(w * 2 ** 20),
                           activation_bytes=int(a * 2 ** 20), overhead_bytes=0)


def test_a_component_no_tiling_fits_declines_the_rung_and_the_refusal_names_it():
    """The supervisor's decision (2026-09-29 11:33): under Triton a host rung whose component the card
    cannot hold, whole or tiled, is never planned — it fails there; the refusal carries the arithmetic
    and the engine that runs it (`--compiled`, on the host)."""
    s, card = _stub()
    comps = [("vae", _m(300, 60_000.0)), ("transformer", _m(3_000, 32_000.0)), ("text_encoder", _m(9_000, 200.0))]
    assert s._size_host_placement_on_device("cpu_streaming", comps, [card], None) is False
    assert set(s._host_device_tilings["cpu_streaming"]) == {"vae"}
    assert s._host_device_overflow["cpu_streaming"] == ("cuda:0", 14_000.0, {"transformer": 32_000.0})
    s._strategies_tried, s._layer_streaming_declined = ["cpu_streaming"], None
    import pytest
    with pytest.raises(RuntimeError) as exc:
        s._fail_error(comps, [card])
    msg = str(exc.value)
    assert ("transformer's activations (32,000 MB untiled; the tiling engine returned no tile)" in msg
            and "--compiled" in msg), msg


def test_the_compiled_engine_is_not_sized_on_the_card():
    s, card = _stub("compiled")
    assert s._size_host_placement_on_device("cpu_streaming", [("vae", _m(300, 60_000.0))], [card], None) is True
    assert "cpu_streaming" not in s._host_device_tilings, "the compiled engine computes on the host"


def test_a_unified_card_counts_the_weights_it_shares(monkeypatch):
    """On unified memory the host IS the card: a component's weights share the pool its activations
    use, so the whole figure is the component's (review 2026-09-29, R23)."""
    from neurobrix.core.prism import solver as S
    s, card = _stub()
    monkeypatch.setattr(S, "_device_is_unified", lambda dev, profile: True)
    comps = [("text_encoder", _m(9_000, 6_000.0))]           # activations fit 14 000, weights + activations do not
    assert s._size_host_placement_on_device("cpu_streaming", comps, [card], None, object()) is False
    monkeypatch.setattr(S, "_device_is_unified", lambda dev, profile: False)
    assert s._size_host_placement_on_device("cpu_streaming", comps, [card], None, object()) is True


def test_a_component_placement_never_puts_under_triton_on_the_host_what_the_card_cannot_hold():
    """Strategy 4 of a component placement (`_place_component`): under Triton a `cpu` placement computes
    on the card, and a component reaching it fits the card neither whole nor tiled — declined and
    recorded for the refusal; the compiled engine keeps it on the host. Wan2.1-VACE 720x1280 fell to
    lazy_sequential with its transformer there once the host rungs declined (2026-09-29)."""
    from types import SimpleNamespace
    from neurobrix.core.prism import solver as S
    s, card = _stub()
    card.get_cost_multiplier = lambda dt: 1.0
    card.free_mb = 16000
    s._get_component_dtype = lambda c, n: "float16"
    s._whole_component_mb = lambda c, n, m, d: m.total_mb
    s._place_component_fgp = lambda *a, **k: None
    s._spatial_component_tiling = lambda *a, **k: None
    prof = SimpleNamespace(cpu=SimpleNamespace(ram_mb=257_530), devices=[])
    s._host_budget_mb = lambda p: 200_000
    orig = S._device_is_unified
    S._device_is_unified = lambda dev, profile: False
    try:
        mem = _m(3_000, 32_000.0)
        assert s._place_component(None, "transformer", mem, [card], {}, prof) is None
        assert s._host_device_overflow["a component's host placement"] == ("cuda:0", 14_000.0, {"transformer": 32_000.0})
        s._mode = "compiled"
        assert s._place_component(None, "transformer", mem, [card], {}, prof)[0] == "cpu"
    finally:
        S._device_is_unified = orig
