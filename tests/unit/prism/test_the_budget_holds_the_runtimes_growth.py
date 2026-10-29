"""A card's budget holds room for what the engine's runtime grows to during a run beyond the context read when the plan
starts (the kernels' modules and their local memory, outside every allocator): the vendor file's measured
`memory.<engine>_runtime_growth_mb`, taken off the budget and stated in its note; unmeasured, nothing is taken and the
note says so. Lot gate 2026-10-28 on V100-16GB: Wan2.2-I2V-A14B and Sana 4K triton held 1.0-1.2 GB outside the allocator
for a 306-312 MB context and their pieces, cut to the card less the context alone, ran out of memory. No card."""
from neurobrix.core.prism import PrismSolver
from tests.unit.prism._pinned_machine import V100_16GB, pin_dedicated_card, pin_host, profile
from tests.unit.prism.test_a_component_over_the_rung_streams_inside_itself import no_door


def _budget(monkeypatch, growth):
    import neurobrix.core.config.loader as L
    real = L.get_vendor_config

    def vendor(brand, arch):
        cfg = dict(real(brand, arch))
        mem = dict(cfg.get("memory") or {})
        mem["triton_runtime_growth_mb"] = growth
        cfg["memory"] = mem
        return cfg
    monkeypatch.setattr(L, "get_vendor_config", vendor)
    no_door(monkeypatch)
    pin_host(monkeypatch, 257530, 200000, "an idle rack host")
    pin_dedicated_card(monkeypatch, 16151, 306, "an idle V100-16GB")
    s = PrismSolver()
    s._mode = "triton"
    dev, = [d for d in s._prepare_devices(profile(V100_16GB)) if d.device_string.startswith("cuda")]
    return dev


def test_the_measured_growth_comes_off_the_budget_and_is_said(monkeypatch):
    base = _budget(monkeypatch, None)
    grown = _budget(monkeypatch, 900)
    assert base.budget_mb - grown.budget_mb == 900, (base.budget_mb, grown.budget_mb)
    assert "triton_runtime_growth_mb" in grown.budget_note and "900 MB" in grown.budget_note, grown.budget_note
    assert "not measured" in base.budget_note, base.budget_note


def _vendor_growth(monkeypatch, growth):
    import neurobrix.core.config.loader as L
    real = L.get_vendor_config

    def vendor(brand, arch):
        cfg = dict(real(brand, arch))
        cfg["memory"] = {**(cfg.get("memory") or {}), "triton_runtime_growth_mb": growth}
        return cfg
    monkeypatch.setattr(L, "get_vendor_config", vendor)


def test_the_whole_rung_is_one_components_only_where_the_growth_is_measured(monkeypatch):
    """Under Triton a component held whole may take the whole rung where the budget holds the runtime's measured growth;
    where the growth is not measured the whole-component fraction stands — dropped there, the margin it held would be
    priced by nothing (the doctrine review of 2026-10-28). The compiled engine keeps the fraction."""
    for growth, mode, whole in ((900, "triton", True), (None, "triton", False), (900, "compiled", False)):
        _vendor_growth(monkeypatch, growth)
        no_door(monkeypatch)
        pin_host(monkeypatch, 257530, 200000, "an idle rack host")
        pin_dedicated_card(monkeypatch, 16151, 306, "an idle V100-16GB")
        s = PrismSolver()
        s._mode = mode
        dev, = [d for d in s._prepare_devices(profile(V100_16GB)) if d.device_string.startswith("cuda")]
        want = s._effective_capacity_mb(dev) * (1.0 if whole else s.whole_component_fraction)
        assert abs(s._rung_part_mb(dev) - want) < 1e-6, (growth, mode, s._rung_part_mb(dev), want)


def test_a_unified_rung_holds_the_loaders_copies_in_flight(monkeypatch):
    """On a unified device the Triton loader's copies in flight sit in the pool beside the device plan while it loads:
    the rung's part is the rung less them (deepseek-moe-16b-chat on the M4 Pro, 2026-10-28: an expert cache sized to the
    whole rung put the plan's host side 102 MB over the pool). Read outside a solve, the copies are refused by name."""
    import pytest
    from tests.unit.prism._pinned_machine import APPLE_M4_PRO
    _vendor_growth(monkeypatch, 500)
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 17510, "the Mac at 17.1 GB free")
    s = PrismSolver()
    s._mode = "triton"
    dev = s._prepare_devices(profile(APPLE_M4_PRO))[0]
    with pytest.raises(RuntimeError, match="outside a solve"):
        s._rung_part_mb(dev)
    s._load_copies_mb = 800.0
    assert abs(s._rung_part_mb(dev) - (s._effective_capacity_mb(dev) - 800.0)) < 1e-6
