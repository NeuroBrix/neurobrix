"""A device's planning capacity is its DECLARED figure; no coefficient hedges it.

`safety_margin` (0.95, config/system.yml) multiplied every capacity before the budget law and its measured reserves:
the Mac's pool read 17 277 MB where the device declares 18 186 (the Mac, 2026-10-25 18:10 — the 909 MB no fact
explained in the compiled room), a V100-16GB 15 565 where it holds 16 384. The budget is the declared figure less
MEASURED facts (the owner, 15:16): the runtime's own context, the engine's measured base, the vendor's reserve.

What this test would do if the code were wrong: the two capacities read 95 % of the declared figures and both
assertions fail. Seen red on b6d36c2b (17 277 / 15 565), green with the coefficient removed."""
from neurobrix.core.prism import PrismSolver
from tests.unit.prism._pinned_machine import APPLE_M4_PRO, V100_16GB, pin_dedicated_card, pin_host, profile


def test_a_unified_pool_plans_at_its_declared_figure_less_the_engines_measured_base(monkeypatch):
    pin_host(monkeypatch, 24576, 18186, "the Mac, idle")
    s = PrismSolver(); s._mode = "triton"
    dev = s._prepare_devices(profile(APPLE_M4_PRO))[0]
    base = (APPLE_M4_PRO.get("cpu", {}).get("runtime_base_mb") or {}).get("triton", 0)
    assert dev.recommended_mb == 18186 and dev.capacity_mb == 18186 - base, (dev.recommended_mb, dev.capacity_mb, base)


def test_a_dedicated_card_plans_at_its_declared_figure_less_its_own_context(monkeypatch):
    pin_dedicated_card(monkeypatch, 16151, 306, "an idle V100-16GB")
    pin_host(monkeypatch, 257530, 200000, "an idle rack host")
    dev = PrismSolver()._prepare_devices(profile(V100_16GB))[0]
    # the law: the card's figure (the driver's total, 16 151 on this SKU) less the runtime's own context (306)
    assert dev.recommended_mb == 16384 and dev.capacity_mb == 16151 - 306, (dev.recommended_mb, dev.capacity_mb)
    assert dev.capacity_mb == dev.budget_mb, "one figure every rung plans against"


def test_under_the_census_door_every_figure_a_rung_reads_is_the_rung(monkeypatch):
    """CogVideoX-5b's 13 455 MB transformer was held whole on an imposed 8 192 MB rung (2026-10-25): the rungs
    read `capacity_mb` / `free_mb` (the declared pool) while only `budget_mb` carried the rung."""
    pin_host(monkeypatch, 24576, 18186, "the Mac, idle")
    monkeypatch.setenv("NBX_PRISM_BUDGET_MB", "8192")
    dev = PrismSolver()._prepare_devices(profile(APPLE_M4_PRO))[0]
    assert dev.capacity_mb == dev.budget_mb == dev.free_mb == 8192, (dev.capacity_mb, dev.budget_mb, dev.free_mb)
    assert dev.recommended_mb == 18186
