"""A card this process has not touched yet holds no context of ours; the run makes one there the moment it places
anything on it. The 4-card pass (2026-10-27, profile c4140-4xv100-16GB-nvlink) read three untouched V100-16GB at their
whole 16 151 MB and planned Qwen3-30B-A3B-Thinking-2507 triton as weight_sharding at 15 867 MB per card, where the
census shadow — reading each card as a run reads it, total less the profile's measured context — budgets 15 845 MB
and plans lazy_sequential: the coverage check passed a plan the run did not make, and the run died on the cards.
The live reading now counts the profile's runtime context on a card that has none of ours yet.

What the test would do if the code were wrong: the untouched card's budget is its whole total (red); the plan of
the live 4-card machine is weight_sharding, not the shadow's (red). No card."""
from __future__ import annotations

from neurobrix.core.prism import solver as solver_mod
from neurobrix.core.prism.memory_budget import DeviceReading, budget_mb
from tests.unit.prism._pinned_machine import pin_host

PROFILE = "c4140-4xv100-16GB-nvlink"


def _untouched_cards(monkeypatch, total_mb=16151):
    from nbxtensor.tensor import DeviceAllocator
    monkeypatch.setattr(solver_mod, "read_device_sharing", lambda _i: DeviceReading(
        kind="device", capacity_mb=total_mb, free_mb=total_mb, own_context_mb=0.0, measured=True,
        source="injected: an idle card this process has not touched"))
    monkeypatch.setattr(DeviceAllocator, "free_memory_mb", staticmethod(lambda _i: total_mb))


def _profile():
    from neurobrix.core.prism.loader import load_profile
    return load_profile(PROFILE)


def test_an_untouched_card_is_budgeted_less_the_runtime_context(monkeypatch):
    _untouched_cards(monkeypatch)
    prof = _profile()
    dev = prof.devices[1]
    r = solver_mod.PrismSolver()._device_reading(dev, float(dev.memory_mb), None, prof)
    assert budget_mb(r) == 16151 - int(dev.runtime_context_mb), (budget_mb(r), r)
    assert r.free_mb == 16151 - float(dev.runtime_context_mb)


def test_the_live_four_card_plan_is_the_shadow_s(monkeypatch):
    from tests.unit.prism.test_a_component_over_the_rung_streams_inside_itself import _input_config, no_door
    no_door(monkeypatch)
    pin_host(monkeypatch, 257530, 200000, "an idle rack host")
    _untouched_cards(monkeypatch)
    c, ic = _input_config("Qwen3-30B-A3B-Thinking-2507", "triton")
    p = solver_mod.PrismSolver().solve_smart(c, _profile(), ic, mode="triton")
    assert p.strategy == "lazy_sequential", p.strategy
