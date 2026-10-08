"""On unified memory a plan whose host side overshoots the free reading tries the next strategy at the
SAME rung before the rung steps down.

The Mac, 2026-10-08 01:43, native (compiled) pass: VibeVoice-1.5B read 14 930 MB free. single_gpu loads
eagerly, so its host side (15 066 MB: every component's read and pinned copy, plus the device plan) is
the same at every rung — lowering the rung shrinks the device window, never the eager load. The solver
re-solved single_gpu at 12 288, 11 264, 8 192 and 6 144 MB with the same 15 066 MB, then took
layer_streaming at 4 096, which reloaded the language model's segments every step: 3 850 reloads, a
2 400 s timeout. At 15 800 MB free the same container plans single_gpu. A lazy strategy at the top rung
holds one component's load at a time, which fits the reading. Reproduced on the Mac's profile
(_pinned_machine.APPLE_M4_PRO) with the host reader set to the run's reading; no card."""
from neurobrix.core.prism import InputConfig, PrismSolver
from neurobrix.nbx import NBXContainer
from tests.unit.prism._pinned_machine import APPLE_M4_PRO, container_root, pin_host, profile

MODEL, FREE = "VibeVoice-1.5B", 14930


def test_the_rung_steps_down_only_when_no_strategy_at_it_fits_the_host_side(monkeypatch):
    pin_host(monkeypatch, 24576, FREE, "the Mac, 2026-10-08 01:43")
    c = NBXContainer.load(str(container_root(MODEL)))
    p = PrismSolver().solve_smart(c, profile(APPLE_M4_PRO), InputConfig(batch_size=1), mode="compiled")
    fp = p.host_footprint
    host_side = (fp["total_bytes"] - fp["resident_bytes"]) / (1 << 20)
    top = p.unified_rungs_tried[0][0]
    assert host_side <= FREE, (p.strategy, int(host_side), p.unified_rungs_tried)
    assert p.strategy != "layer_streaming", p.unified_rungs_tried
    assert {r for r, _, _ in p.unified_rungs_tried} == {top}, p.unified_rungs_tried
