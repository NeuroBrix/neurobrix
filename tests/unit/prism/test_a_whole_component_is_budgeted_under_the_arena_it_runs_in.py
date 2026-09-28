"""A component held WHOLE is budgeted under the memory model the run executes under.

mochi-1-preview's VAE at 85 frames, 320x576: Prism profiled its activation at 24 561 MB, placed
it whole on a 32 GB V100 (346 MB of weights beside it, the whole figure under the 0.92 usable
fraction) and the decode died at `aten.silu::32` — 25 202 MB live, 8 493 MB asked, at least
33 695 MB against a 32 501 MB card (2026-09-25, card 2; `validation_outputs/rebuilds_2026_09_25/
mochi-1-preview/f85_320x576.log`). At 480x848 the profiled 54 179 MB crossed the usable budget,
the VAE was TILED (9 147 MB tiled) and decoded. The triton arena's live watermark runs ~1.3x the
profiled peak — the tiling call site says so and budgets TILES by it; the whole-component test
compared the bare profiled peak against the card.

Now `_whole_component_mb` applies `PRISM_DEFAULTS["triton_arena_activation_factor"]` to the
activation for the triton modes: this request receives a component-tiling spec for its VAE under
`triton`. Seen RED on main (776a6c6d): `component_tiling == {}` under triton — the plan the card
refused. (Since 86aa0d87 the VAE is priced at the width it executes and is tiled under `compiled`
too — see the last cell.)

    PYTHONPATH=src pytest tests/unit/prism/test_a_whole_component_is_budgeted_under_the_arena_it_runs_in.py -p no:cacheprovider
"""
from __future__ import annotations

import pytest

from neurobrix.core.prism.profiler import InputConfig
from neurobrix.core.prism.solver import PrismSolver
from neurobrix.nbx.container import NBXContainer
from tests.unit.prism._pinned_machine import (V100_16GB, container_root, impose_rung,
                                              pin_dedicated_card, profile)

V100_32GB = {**V100_16GB, "id": "scenario-v100-32gb",
             "devices": [{**V100_16GB["devices"][0], "model": "Tesla V100-SXM2-32GB", "memory_mb": 32768}]}
def _request():
    """The request as the CLI forms it: the temporal compression is the container's own
    (`runtime/defaults.json`: `temporal_compression_ratio`), never assumed."""
    import json
    dj = json.loads((container_root("mochi-1-preview") / "runtime" / "defaults.json").read_text())
    # the CFG pair (positive + negative prompt) the run batches when the container's guidance
    # scale asks for guidance — the CLI's plan carried batch 2 (VAE activation 24 561 MB, not
    # the 12 280 MB of a single sample)
    batch = 2 if float(dj.get("guidance_scale", 1.0)) > 1.0 else 1
    return dict(batch_size=batch, num_frames=85, height=320, width=576,
                temporal_compression=int(dj["temporal_compression_ratio"]))


def _plan(monkeypatch, mode):
    pin_dedicated_card(monkeypatch, 32501, 267, "the rack's card 2, 2026-09-25")
    impose_rung(monkeypatch, 32768)
    monkeypatch.delenv("NBX_FORCE_STRATEGY", raising=False)
    c = NBXContainer.load(str(container_root("mochi-1-preview")))
    return PrismSolver().solve_smart(c, profile(V100_32GB), InputConfig(**_request()), mode=mode)


def test_the_vae_that_died_whole_under_triton_is_tiled_by_the_plan(monkeypatch):
    plan = _plan(monkeypatch, "triton")
    tiling = getattr(plan, "component_tiling", None) or {}
    assert "vae" in tiling, f"the VAE is placed whole again — the plan the card refused: {tiling}"
    assert tiling["vae"]["tiled_activation_bytes"] < 24_561 * 2 ** 20, tiling["vae"]


def test_the_vae_is_planned_at_the_width_the_card_measured(monkeypatch):
    """CHANGED 2026-09-28 (86aa0d87, the estimator prices the width each engine executes). This cell
    was `test_the_compiled_engine_keeps_the_profiled_figure` and asserted the compiled plan does NOT
    tile the VAE — a proxy for the old figure, priced with every float at the compute dtype
    (24 561 MB). The figure itself is asserted now.

    The anchor is the card: the Triton run of this request died at `aten.silu::32` asking
    8 493 465 600 bytes for shape (1, 128, 90, 320, 576) — fp32, 8 100 MiB
    (validation_outputs/rebuilds_2026_09_25/mochi-1-preview/f85_320x576.log:118, card 2,
    2026-09-25, Engine: TRITON). Main priced that tensor at 4 050 MiB. The width pass prices it at
    the measured bytes, and the plan's VAE activation is the profiler's peak at those widths, put
    through the solver's own request scaling: 49 121 MB, over the usable 32 GB — so it is tiled
    under compiled as well.

    The COMPILED half is INFERRED from the ATen engine's table (group_norm is AMP_FP32 without a
    calibration record, the silu that follows keeps its input's width); no card has run this
    request under the compiled engine. That measurement is owed."""
    import math
    from neurobrix.core.prism.profiler import ActivationProfiler
    from neurobrix.core.prism.runtime_widths import plan_time_contract, runtime_widths
    pin_dedicated_card(monkeypatch, 32501, 267, "the rack's card 2, 2026-09-25")
    impose_rung(monkeypatch, 32768)
    monkeypatch.delenv("NBX_FORCE_STRATEGY", raising=False)
    root = container_root("mochi-1-preview")
    c = NBXContainer.load(str(root))
    s = PrismSolver()
    seen = {}
    real = s._compute_memory

    def spy(*a, **k):
        out = real(*a, **k)
        seen.update(out)
        return out

    s._compute_memory = spy
    # the solver completes the request it plans (the VAE's scale, from the container): the same
    # object is read back below
    request = InputConfig(**_request())
    plan = s.solve_smart(c, profile(V100_32GB), request, mode="compiled")

    comp = next(x for x in c.get_neural_components() if x.name == "vae")
    g = comp.graph
    prof = ActivationProfiler(g)
    smap = prof.build_symbol_map(request, placement_floor=True)
    widths = runtime_widths(g, "float16", "compiled", has_native_bf16=False,
                            contract=plan_time_contract(root, "vae", g, "float16"),
                            shape_of=lambda t: prof._resolve_shape(g["tensors"][t], smap))
    silu = g["ops"]["aten.silu::32"]["output_tensor_ids"][0]
    numel = math.prod(prof._resolve_shape(g["tensors"][silu], smap))
    assert numel * widths[silu] == 8_493_465_600, (numel, widths[silu])       # the card's malloc

    peak = prof.estimate_peak_memory(request, dtype_bytes=2, widths=widths,
                                     placement_floor=True).peak_bytes
    assert seen["vae"].activation_bytes == s._scale_activations_to_request(comp, peak, request), (
        seen["vae"].activation_bytes / 2 ** 20, peak / 2 ** 20)
    assert "vae" in (getattr(plan, "component_tiling", None) or {}), plan.component_tiling


def test_the_factor_is_data_with_its_provenance():
    from neurobrix.core.config.system import get_prism_defaults
    d = get_prism_defaults()
    assert 1.0 < float(d["triton_arena_activation_factor"]) <= 2.0, d
