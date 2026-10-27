"""A rung lowered below the pool's facts — the unified descent (`_unified_rung_cap_mb`) or a busy machine re-planned on
what is free now (`_busy_caps`, `solve_smart`) — lowers the rung a tiled or streamed component sizes its pieces from
(`tile_rung_mb`), not only the budget. Before (the Mac, 2026-10-27 01:21): `tile_rung_mb` stayed the device's nominal
rung (16 384 MB on the M4 Pro) whatever the cap, so mochi-1-preview's VAE kept its temporal chunks [1,2,2] (13 089 MB)
down to the 4 096 MB rung, the busy re-plan priced the same 15 276 MB host side as the facts' plan and refused — where
1-frame chunks (10 159 MB) fit the 14 218 MB then free and ran (24/10, 26/10 23:4x).
"""
import pytest

from neurobrix.core import host_memory as HM
from neurobrix.core.prism import solver as S
from neurobrix.core.prism.memory_budget import rung_down_mb
from tests.unit.prism._pinned_machine import APPLE_M4_PRO, profile


def _solver(monkeypatch):
    monkeypatch.setattr(S, "memory_state", lambda: HM.MemoryState(available_mb=17582, source="the Mac's reading"))
    monkeypatch.setattr(S, "_census_shadow_active", lambda: False)
    return S.PrismSolver()


def test_the_facts_rung_is_the_device_s_own(monkeypatch):
    dev = _solver(monkeypatch)._prepare_devices(profile(APPLE_M4_PRO))[0]
    assert dev.tile_rung_mb == rung_down_mb(18186)               # untouched where nothing lowers the rung


@pytest.mark.parametrize("cap", [12288, 6144, 4096])
def test_the_unified_descent_lowers_the_tile_rung(monkeypatch, cap):
    s = _solver(monkeypatch)
    s._unified_rung_cap_mb = float(cap)
    dev = s._prepare_devices(profile(APPLE_M4_PRO))[0]
    assert dev.tile_rung_mb <= cap, (cap, dev.tile_rung_mb)


def test_a_busy_re_plan_lowers_the_tile_rung(monkeypatch):
    s = _solver(monkeypatch)
    s._busy_caps = {"mps:0": 12500.0}
    dev = s._prepare_devices(profile(APPLE_M4_PRO))[0]
    assert dev.tile_rung_mb <= rung_down_mb(12500.0), dev.tile_rung_mb
