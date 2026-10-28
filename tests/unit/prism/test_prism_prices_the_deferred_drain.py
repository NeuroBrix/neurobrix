"""Prism prices the Triton engine's deferred drain: the plan leaves room for the dead-tensor queue.

The Triton hot loops (`triton/sequence.py`) retire every dead or overwritten tensor onto a queue and
free it, after a device sync, only when the queue reaches the drain cliff — the vendor profile's
`memory.deferred_drain_cliff_bytes`. Under the allocator pool (the default) no earlier drain fires,
so between two drains up to a cliff of dead bytes sits on the card beside whatever runs. The plan
priced the live set and not that queue, and the allocator rescued the difference by draining on a
refused malloc and retrying: `[NBX_OOM_RECLAIM] freed ...` 1 279 times in Allegro's 16 GB triton
run of the mq24 gate (2026-10-10), up to 1 883 MB a time in CogVideoX-5b-I2V's. The rescue is gone
(`tests/unit/kernels/test_a_refused_malloc_is_raised_not_rescued.py`); the price replaces it.

The plan here is SYNTHETIC: one component built from the card's own figures — its whole cost set
half a drain under the usable part of the rung before the drain is priced — so it fits only when
the queue is not counted. The figures are read from the solver and the vendor profile, never
written here.

Option B (merge-queue-25): the plan carries a per-device cliff, `min(profile cliff, room)`, room =
the rung's part less the largest component it holds whole there; a component is held whole only if
that room is at least the profile's `deferred_drain_floor_bytes`, and the hot loop drains at the
plan's cliff.

What these cells would do on the code before the price: the first fails at the missing
`_deferred_drain_mb`; with the price forced to 0 (the injection), the first fails on the figure and
the floor cell places the component whole on the card under Triton — the plan that needed the
rescue. With the plan's cliff not recorded or not read, the room and hot-loop cells fail.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from neurobrix.core.config.loader import get_vendor_config
from neurobrix.core.prism.solver import PrismSolver
from tests.unit.prism._pinned_machine import V100_16GB, impose_rung, pin_dedicated_card, profile

MB = 1024 * 1024


def _card(monkeypatch, mode):
    pin_dedicated_card(monkeypatch, 16151, 267, "the rack's card 0, 2026-09-29")
    impose_rung(monkeypatch, 16384)
    monkeypatch.delenv("NBX_DEFERRED_DRAIN_BYTES", raising=False)
    s = PrismSolver()
    s._mode = mode
    dev = s._prepare_devices(profile(V100_16GB))[0]
    return s, dev


def _cliff_mb(dev) -> float:
    brand = getattr(dev.spec.brand, "value", dev.spec.brand)
    return get_vendor_config(brand, dev.spec.architecture)["memory"]["deferred_drain_cliff_bytes"] / MB


def test_the_usable_part_of_a_card_is_net_of_the_drain_under_triton_only(monkeypatch):
    s_c, dev_c = _card(monkeypatch, "compiled")
    s_t, dev_t = _card(monkeypatch, "triton")
    cliff = _cliff_mb(dev_t)
    assert cliff > 0
    assert s_c._deferred_drain_mb(dev_c) == 0.0, "the compiled engine has no deferred queue to price"
    assert s_t._deferred_drain_mb(dev_t) == pytest.approx(cliff)
    # each engine's budget holds its own runtime's growth (`PrismSolver._runtime_growth`: the Triton one measured on
    # this device, the compiled one not): the drain is what the Triton part leaves out of ITS rung part
    assert s_t._usable_mb(dev_t) == pytest.approx(s_t._rung_part_mb(dev_t) - cliff), (
        "the Triton plan does not leave the drain cliff out of what one component may hold")
    assert s_c._usable_mb(dev_c) == pytest.approx(s_c._rung_part_mb(dev_c))


def test_the_door_that_pins_the_runs_cliff_pins_its_price(monkeypatch):
    s, dev = _card(monkeypatch, "triton")
    monkeypatch.setenv("NBX_DEFERRED_DRAIN_BYTES", str(512 * MB))
    assert s._deferred_drain_mb(dev) == pytest.approx(512.0)


def _place(s, dev, mem):
    s._get_component_dtype = lambda c, n: "float16"
    s._place_component_fgp = lambda *a, **k: None
    s._spatial_component_tiling = lambda *a, **k: None
    return s._place_component(None, "denoiser", mem, [dev], {}, profile(V100_16GB))


def _floor_mb(dev) -> float:
    brand = getattr(dev.spec.brand, "value", dev.spec.brand)
    return get_vendor_config(brand, dev.spec.architecture)["memory"]["deferred_drain_floor_bytes"] / MB


def _component_of_whole(s_t, dev_t, whole_target):
    """A synthetic component whose whole cost on `dev_t` is `whole_target` MB (the arena factor
    read from the solver, never written here)."""
    factor = s_t._live_activation_mb(SimpleNamespace(activation_mb=1.0))
    act = 1024.0
    weight = whole_target - act * factor
    mem = SimpleNamespace(weight_mb=weight, activation_mb=act, total_mb=weight + act,
                          weight_bytes=int(weight * MB), activation_bytes=int(act * MB),
                          overhead_bytes=0, peak_op_uid=None, activation_profiled=True)
    s_t._get_component_dtype = lambda c, n: "float16"
    assert s_t._whole_component_mb(None, "denoiser", mem, dev_t) == pytest.approx(whole_target)
    return mem


def _plan_cliff(s, dev, mem, placed_on) -> int:
    """The cliff `_assign_drain_cliffs` sets on a one-component plan placed on `placed_on`."""
    from neurobrix.core.prism.solver import ComponentAllocation
    alloc = ComponentAllocation(name="denoiser", devices=[placed_on.split("zero3:")[-1]],
                                dtype="float16", memory_mb=mem.total_mb, architecture="volta",
                                vendor="nvidia",
                                strategy="zero3" if placed_on.startswith("zero3:") else "single_gpu")
    plan = SimpleNamespace(components={"denoiser": alloc}, component_memory={"denoiser": mem},
                           layer_stream_plan={}, component_tiling={}, deferred_drain_cliffs={})
    s._assign_drain_cliffs(None, plan, [dev])
    assert plan.deferred_drain_cliffs.get(dev.device_string) == alloc.deferred_drain_cliff_bytes
    return alloc.deferred_drain_cliff_bytes


def test_a_component_whole_beside_a_smaller_queue_stays_whole_and_the_plan_drains_at_its_room(monkeypatch):
    """Option B (merge-queue-25): a component that fits whole beside a queue SMALLER than the
    profile's cliff, yet at least the profile's floor, is held whole, and the plan records the
    room it leaves as that device's cliff — the Janus-on-the-idle-Mac shape (room 904 MB)."""
    s_c, dev_c = _card(monkeypatch, "compiled")
    s_t, dev_t = _card(monkeypatch, "triton")
    cliff, floor = _cliff_mb(dev_t), _floor_mb(dev_t)
    part = s_t._rung_part_mb(dev_t)                     # the Triton rung's part (its budget holds its runtime's growth)
    assert s_c._usable_mb(dev_c) == pytest.approx(s_c._rung_part_mb(dev_c))   # no queue under compiled
    room = (cliff + floor) / 2                          # between the floor and the profile's cliff
    mem = _component_of_whole(s_t, dev_t, part - room)
    placed = _place(s_t, dev_t, mem)
    assert placed is not None and placed[0] == dev_t.device_string, (
        f"a component leaving {room:,.0f} MB (floor {floor:,.0f}, cliff {cliff:,.0f}) not held whole: {placed}")
    planned = _plan_cliff(s_t, dev_t, mem, placed[0])
    assert planned == pytest.approx(room * MB, abs=MB), (
        f"the plan drains at {planned / MB:,.0f} MB, not the {room:,.0f} MB the component leaves")
    assert planned < cliff * MB
    # The compiled engine has no queue: the same component is whole, and no cliff is recorded.
    assert _place(s_c, dev_c, mem)[0] == dev_c.device_string
    assert _plan_cliff(s_c, dev_c, mem, dev_c.device_string) is None


def test_a_component_leaving_less_than_the_floor_is_not_held_whole_under_triton(monkeypatch):
    s_c, dev_c = _card(monkeypatch, "compiled")
    s_t, dev_t = _card(monkeypatch, "triton")
    floor = _floor_mb(dev_t)
    part = s_c._usable_mb(dev_c)
    mem = _component_of_whole(s_t, dev_t, part - floor / 2)     # leaves half the floor
    placed = _place(s_t, dev_t, mem)
    assert placed is not None
    assert placed[0] == f"zero3:{dev_t.device_string}", (
        f"a component leaving {floor / 2:,.0f} MB beside it, under the {floor:,.0f} MB drain floor, "
        f"placed {placed[0]} — the plan the allocator had to rescue")
    # Its plan, held off the card, drains at the profile's cliff.
    assert _plan_cliff(s_t, dev_t, mem, placed[0]) == int(_cliff_mb(dev_t) * MB)
    # The control: the compiled engine has no queue, and holds the same component whole.
    assert _place(s_c, dev_c, mem)[0] == dev_c.device_string


def test_the_hot_loop_drains_at_the_plans_cliff_and_the_door_still_pins_it(monkeypatch):
    from neurobrix.kernels.wrappers import deferred_drain_policy
    from neurobrix.triton.sequence import TritonSequence, _resolve_drain_policy

    monkeypatch.delenv("NBX_DEFERRED_DRAIN_BYTES", raising=False)
    monkeypatch.delenv("NBX_DEFERRED_DRAIN_RESERVE", raising=False)
    prof_cliff, prof_floor = deferred_drain_policy()[:2]
    planned = 904 * MB
    assert planned < prof_cliff
    cliff, floor = _resolve_drain_policy(planned)[:2]
    assert cliff == planned, "the hot loop drains at the profile's cliff, not the plan's"
    assert floor == min(prof_floor, planned)
    assert _resolve_drain_policy(None)[:2] == (prof_cliff, prof_floor)
    # The sequence carries the plan's figure to both hot loops (and to its sub-sequences).
    seq = TritonSequence({"ops": {}, "tensors": {}}, stores_fp64=False, drain_cliff_bytes=planned)
    assert seq._planned_drain_cliff == planned
    monkeypatch.setenv("NBX_DEFERRED_DRAIN_BYTES", str(300 * MB))
    assert _resolve_drain_policy(planned)[:2] == (300 * MB, 300 * MB), "the door no longer pins the cliff"
