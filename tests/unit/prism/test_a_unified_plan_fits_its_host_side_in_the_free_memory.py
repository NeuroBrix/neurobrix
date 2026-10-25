"""On unified memory the device plan and the run's host side come out of ONE pool, and since 2026-10-25 (the owner,
15:16) that pool is the profile's DECLARED one — the installed RAM less the engine's measured base — never the free
reading of the moment: the plan is the same under every reading, and a reading too low for it refuses at the door,
naming the pool. Before: the Mac, 2026-10-03 23:48, Janus planned a 7 531 MB window at 11 638 MB free and 10 362 MB
at 12 652 MB free; 2026-10-25 16:56, HAT-L (157 MB of weights) planned layer_streaming beside another run and
single_gpu a minute later. Reproduced on the Mac's profile (_pinned_machine.APPLE_M4_PRO); no card."""
import pytest
from neurobrix.core.prism import InputConfig, PrismSolver
from neurobrix.nbx import NBXContainer
from tests.unit.prism._pinned_machine import APPLE_M4_PRO, container_root, pin_host, profile
CASES = {"Janus-Pro-7B": 384, "Flex.1-alpha": 1024}     # each model's request extent (its CLI default)
READINGS = (10800, 11638, 12652, 20000)                  # the Mac's free readings that evening, and an idle machine


def _plan(monkeypatch, model, free):
    pin_host(monkeypatch, 24576, free, "the Mac, 2026-10-03")
    c = NBXContainer.load(str(container_root(model)))     # absent: a skip that names it
    hw = CASES[model]
    return PrismSolver().solve_smart(c, profile(APPLE_M4_PRO), InputConfig(batch_size=1, height=hw, width=hw),
                                     mode="triton")


@pytest.mark.parametrize("model", sorted(CASES))
def test_the_plan_is_the_same_under_every_free_reading_or_the_door_refuses(monkeypatch, model):
    plans = {}
    for free in READINGS:
        p = _plan(monkeypatch, model, free)                                # the plan is always returned
        fp = p.host_footprint or {}
        need = (int(fp.get("total_bytes", 0)) - int(fp.get("resident_bytes", 0))) / 2**20
        plans[free] = (p.strategy, int(need))
        if need > free:                                                      # the door says so, the plan stands
            assert p.execution_door and "are free now" in p.execution_door and "declared" in p.execution_door, (model, free)
        else:
            assert not p.execution_door, (model, free, p.execution_door)
    assert len(set(plans.values())) == 1, (model, plans)                    # ONE plan, whatever the reading
