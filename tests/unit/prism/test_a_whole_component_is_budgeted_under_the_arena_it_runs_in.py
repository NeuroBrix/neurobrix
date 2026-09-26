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
`triton`, and none under `compiled`, whose allocator the profile measured. Seen RED on main
(776a6c6d): `component_tiling == {}` under triton — the plan the card refused.

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


def test_the_compiled_engine_keeps_the_profiled_figure(monkeypatch):
    """The profile was measured under the compiled engine's allocator; nothing is added to it."""
    plan = _plan(monkeypatch, "compiled")
    tiling = getattr(plan, "component_tiling", None) or {}
    assert "vae" not in tiling, tiling


def test_the_factor_is_data_with_its_provenance():
    from neurobrix.core.config.system import get_prism_defaults
    d = get_prism_defaults()
    assert 1.0 < float(d["triton_arena_activation_factor"]) <= 2.0, d
