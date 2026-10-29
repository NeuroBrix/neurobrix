"""On a unified device the budget is the pool's DECLARED figure (what the device recommends, the profile's
`memory_mb`, less the engine's measured base on this host) — never the free memory of the moment rounded onto the
ladder (the owner, 2026-10-25 15:16: a plan is reproducible from one run to the next; the moment's reading is the
execution door). Janus-Pro-7B on the Mac, 14:56: 0.3 GB less free at a rung edge took layer_streaming, six times slower.

What would this file do if the code were wrong? A budget that follows the reading gives three different figures for
three readings: RED on the first test; the base not taken off: RED on the second.
"""
from types import SimpleNamespace as NS
from pathlib import Path

import pytest

from nbxtensor.backend import host_memory as HM
from neurobrix.core.prism import solver as S
from tests.unit.prism._pinned_machine import APPLE_M4_PRO, profile


def _pool(spec):
    """The declared working set within what the system leaves of the installed RAM (its share measured calm,
    `host_measured.system_share_mb`, 2026-10-26): the one figure both halves of a unified pool read."""
    share = float((getattr(profile(spec), "host_measured", None) or {}).get("system_share_mb") or 0)
    assert share > 0, "the M4 Pro's chip file carries its measured share (2026-10-26)"
    return min(float(spec["devices"][0]["memory_mb"]), float(spec["cpu"]["ram_mb"]) - share)


@pytest.mark.parametrize("free", [11581, 13000, 10380, 17236, 16902])
def test_the_budget_is_the_declared_figure_whatever_is_free(monkeypatch, free):
    st = HM.MemoryState(available_mb=free, source="the Mac's reading")
    monkeypatch.setattr(S, "memory_state", lambda: st)
    monkeypatch.setattr(S, "_census_shadow_active", lambda: False)
    solver = S.PrismSolver()
    devs = solver._prepare_devices(profile(APPLE_M4_PRO))
    declared = _pool(APPLE_M4_PRO)                               # no coefficient since 2026-10-25
    assert devs[0].budget_mb == pytest.approx(declared), (free, devs[0].budget_mb)
    assert devs[0].live_free_mb == free                      # the door carries the moment


def test_the_engine_s_measured_base_comes_off_the_declared_figure(monkeypatch):
    st = HM.MemoryState(available_mb=11581, source="the Mac's reading")
    monkeypatch.setattr(S, "memory_state", lambda: st)
    monkeypatch.setattr(S, "_census_shadow_active", lambda: False)
    import dataclasses
    base = profile(APPLE_M4_PRO)
    def with_base(mb):
        return dataclasses.replace(base, cpu=dataclasses.replace(base.cpu, runtime_base_mb=mb))
    solver = S.PrismSolver()
    solver._mode = "triton"
    declared = _pool(APPLE_M4_PRO)                               # no coefficient since 2026-10-25
    assert solver._prepare_devices(with_base({"triton": 1500}))[0].budget_mb == pytest.approx(declared - 1500)
    assert solver._prepare_devices(with_base({}))[0].budget_mb == pytest.approx(declared)
