"""On unified memory the device plan and the run's host side come out of ONE pool: the rung is the
highest whose WHOLE host side fits the free reading, and a rung no strategy fits steps down the ladder.

The Mac, 2026-10-03 23:48, on its gate tree: Janus-Pro-7B planned a 7 531 MB window (host 12 760 MB) at
11 638 MB free and a 10 362 MB window (host 15 592 MB) at 12 652 MB free — MORE free memory gave a plan
that no longer fit; Flex.1-alpha planned at 10.8 GB free and was REFUSED at 12.6 GB. The rung was read
from the free memory alone. Reproduced on the Mac's profile (_pinned_machine.APPLE_M4_PRO) with the host
reader set to its readings; no card."""
import pytest

from neurobrix.core.prism import InputConfig, PrismSolver
from neurobrix.nbx import NBXContainer
from tests.unit.prism._pinned_machine import APPLE_M4_PRO, container_root, pin_host, profile

CASES = {"Janus-Pro-7B": 384, "Flex.1-alpha": 1024}     # each model's request extent (its CLI default)
READINGS = (10800, 11638, 12652)                         # the Mac's free readings that evening, MB


def _plan(monkeypatch, model, free):
    pin_host(monkeypatch, 24576, free, "the Mac, 2026-10-03")
    try:
        c = NBXContainer.load(str(container_root(model)))
    except Exception as e:                                # a machine without the container
        pytest.skip(f"{model} not in this machine's cache: {e}")
    hw = CASES[model]
    return PrismSolver().solve_smart(c, profile(APPLE_M4_PRO), InputConfig(batch_size=1, height=hw, width=hw),
                                     mode="triton")


@pytest.mark.parametrize("model", sorted(CASES))
def test_more_free_memory_never_gives_a_plan_that_does_not_fit(monkeypatch, model):
    windows = []
    for free in READINGS:
        p = _plan(monkeypatch, model, free)               # never refused: the ladder descends
        fp = p.host_footprint
        host_side = (fp["total_bytes"] - fp["resident_bytes"]) / (1 << 20)
        assert host_side <= free, (model, free, p.strategy, int(host_side), p.unified_rungs_tried)
        windows.append(p.device_window_mb or 0.0)
    assert windows == sorted(windows), (model, list(zip(READINGS, windows)))
