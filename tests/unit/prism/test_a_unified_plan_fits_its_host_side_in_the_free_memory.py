"""On unified memory the device plan and the run's host side come out of ONE pool, and since 2026-10-25 (the owner,
15:16) that pool is the profile's DECLARED one — the installed RAM less the engine's measured base — never the free
reading of the moment: every idle reading gives ONE plan. A reading too low for that plan RE-PLANS on what is free,
said in clear, and the smaller plan's host side fits the reading — never a run that swaps (the owner, 2026-10-26
19:21; before, the door refused it). Before: the Mac, 2026-10-03 23:48, Janus planned a 7 531 MB window at 11 638 MB free and 10 362 MB
at 12 652 MB free; 2026-10-25 16:56, HAT-L (157 MB of weights) planned layer_streaming beside another run and
single_gpu a minute later. Reproduced on the Mac's profile (_pinned_machine.APPLE_M4_PRO); no card."""
import pytest
from neurobrix.core.prism import InputConfig, PrismSolver
from neurobrix.nbx import NBXContainer
from tests.unit.prism._pinned_machine import APPLE_M4_PRO, container_root, pin_host, profile
CASES = {"Janus-Pro-7B": 384, "Flex.1-alpha": 1024}     # each model's request extent (its CLI default)
READINGS = (10800, 11638, 12652, 20000, 24000)           # the Mac's free readings that evening, and two idle ones


def _plan(monkeypatch, model, free):
    pin_host(monkeypatch, 24576, free, "the Mac, 2026-10-03")
    c = NBXContainer.load(str(container_root(model)))     # absent: a skip that names it
    hw = CASES[model]
    return PrismSolver().solve_smart(c, profile(APPLE_M4_PRO), InputConfig(batch_size=1, height=hw, width=hw),
                                     mode="triton")


@pytest.mark.parametrize("model", sorted(CASES))
def test_one_plan_at_rest_and_a_busy_reading_replans_inside_what_is_free(monkeypatch, model):
    """What this test would do if the code were wrong: two idle readings giving two plans, a busy reading left with a
    shut door or a silent re-plan, or a re-plan whose host side does not fit the reading (a run that swaps) — RED."""
    plans, idle = {}, None
    for free in sorted(READINGS, reverse=True):
        p = _plan(monkeypatch, model, free)
        fp = p.host_footprint or {}
        need = (int(fp.get("total_bytes", 0)) - int(fp.get("resident_bytes", 0))) / 2**20
        plans[free] = (p.strategy, int(need), p.busy_replan is not None)
        assert not p.execution_door, (model, free, p.execution_door)
        assert need <= free, (model, free, plans[free])                    # never a run that swaps
        if p.busy_replan is None:
            idle = idle or (p.strategy, int(need))
            assert (p.strategy, int(need)) == idle, (model, plans)          # ONE plan at rest
        else:
            assert idle and p.busy_replan["from"] == idle[0], (model, plans, p.busy_replan)
            assert "machine busy" in p.selection_reason, p.selection_reason
    assert not plans[24000][2] and not plans[20000][2], (model, plans)
