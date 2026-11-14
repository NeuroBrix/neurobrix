"""`replay_graph.CapturedPlan`: a plan recorded on ONE card carries the SETDEV record that names it
(`frozen_plan.SETDEV`, emitted when the device changes), and may carry a device SYNC on that card. Neither makes a graph
node: the plan is captured on that card, one stream orders every node. Until 2026-11-14 the SETDEV fell into the
memset branch ("not enough values to unpack (expected 3, got 1)") and every single-card capture was refused to the direct replay — measured on a V100, TinyLlama with NBX_REPLAY_GRAPH=1: two
refusals, zero graph launches, the same 24 341 copies and 92 877 launches as the direct replay.
What this would do if the code were wrong: the graph replay silently never runs on one card (slower, no error); or a
plan that crosses cards captured as one graph on one card — kernels on the wrong card: wrong bytes, no error."""
import ctypes

import pytest

from nbxtensor.backend import replay_graph as RG
from nbxtensor.backend import runtime as _rt
from nbxtensor.memory.allocator import DeviceAllocator


def test_a_plan_that_crosses_cards_or_syncs_one_is_refused_by_name_before_any_capture():
    with pytest.raises(RG.GraphReplayError, match=r"cards \[0, 1\]"):
        RG.CapturedPlan([("d", (0,)), ("s", (16, 0, 4)), ("d", (1,)), ("s", (32, 0, 4))], None)
    with pytest.raises(RG.GraphReplayError, match=r"cards \[0, 1\]"):
        RG.CapturedPlan([("d", (0,)), ("y", (1,)), ("s", (16, 0, 4))], None)   # a peer copy's wait on its source card
    with pytest.raises(RG.GraphReplayError, match=r"actions \['q'\]"):
        RG.CapturedPlan([("d", (0,)), ("q", (0,)), ("s", (16, 0, 4))], None)
    with pytest.raises(RG.GraphReplayError, match=r"cards \[0, 2\]"):
        RG.CapturedPlan([("d", (2,)), ("s", (16, 0, 4))], 0)       # the slab's card is not the recorded one


@pytest.mark.skipif(_rt.gpu_backend() != "cuda", reason="CUDA graphs: the driver's cuGraph API")
def test_a_plan_carrying_its_own_card_record_is_captured_and_replays_its_copies():
    dev = DeviceAllocator.get_device()
    n = 64
    a, b, c = (DeviceAllocator.malloc_cuda(n) for _ in range(3))
    snap = bytes(range(n))
    plan = RG.CapturedPlan([("d", (dev,)), ("h", (a, snap)), ("y", (dev,)), ("m", (b, a, n, 3)), ("s", (c, 7, n))],
                           None)                          # TinyLlama's decode plan carries a sync on its own card
    try:
        plan.launch()
        got = (ctypes.c_uint8 * n)()
        DeviceAllocator.memcpy(ctypes.addressof(got), b, n, 2)
        assert bytes(got) == snap
        DeviceAllocator.memcpy(ctypes.addressof(got), c, n, 2)
        assert bytes(got) == b"\x07" * n
    finally:
        plan.destroy()
        for p in (a, b, c):
            DeviceAllocator.free_cuda(p)
