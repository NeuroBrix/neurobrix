"""A busy machine is RE-PLANNED on the memory free now, said in clear, never refused while a plan fits (the owner,
2026-10-26 19:21). Until then the execution door refused a plan sized by the pool's facts whenever another application
held part of the pool, even though a slower plan fitted. Both profiles: a V100-16GB another process holds most of, and
the Mac with a browser open. What this test would do if the code were wrong: the plan comes back with its door set and
no re-plan (busy_replan None) — RED; and a machine with no room for any plan must still refuse, naming the shortfall."""
import pytest
from neurobrix.core.prism import InputConfig, PrismSolver
from neurobrix.nbx import NBXContainer
from tests.unit.prism._pinned_machine import (APPLE_M4_PRO, V100_16GB, container_root, no_door, pin_host,
                                              pin_shared_card, profile)

MODEL = "Janus-Pro-7B"


def _solve(spec, mode):
    c = NBXContainer.load(str(container_root(MODEL)))
    return PrismSolver().solve_smart(c, profile(spec), InputConfig(batch_size=1, height=384, width=384), mode=mode)


def _check_busy(p, free_mb):
    assert p.busy_replan is not None, (p.strategy, p.execution_door)
    assert not p.execution_door, p.execution_door
    assert "machine busy" in p.selection_reason and "free now" in p.selection_reason, p.selection_reason
    assert abs(p.busy_replan["free_now_mb"] - free_mb) <= 1, (p.busy_replan, free_mb)


def test_a_shared_v100_is_replanned_on_what_is_free(monkeypatch):
    no_door(monkeypatch)
    pin_host(monkeypatch, 257530, 200000, "an idle rack host")
    pin_shared_card(monkeypatch, 16151, 9000, 306, "a V100-16GB another process holds 9 000 MB of")
    _check_busy(_solve(V100_16GB, "triton"), 16151 - 9000 - 306)


def test_the_mac_with_a_browser_open_is_replanned_on_what_is_free(monkeypatch):
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 9000, "the Mac with ~9 GB free (a browser and apps open)")
    _check_busy(_solve(APPLE_M4_PRO, "triton"), 9000)


def test_no_room_for_any_plan_is_refused_naming_the_shortfall(monkeypatch):
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 300, "the Mac with 300 MB free")
    with pytest.raises(RuntimeError, match="no plan fits|does not fit"):
        _solve(APPLE_M4_PRO, "triton")
