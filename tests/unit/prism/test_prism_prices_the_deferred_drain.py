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

What these cells would do on the code before the price: the first fails at the missing
`_deferred_drain_mb`; with the price forced to 0 (the injection), the first fails on the figure and
the second places the component whole on the card under Triton — the plan that needed the rescue.
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
    assert s_t._usable_mb(dev_t) == pytest.approx(s_c._usable_mb(dev_c) - cliff), (
        "the Triton plan does not leave the drain cliff out of what one component may hold")


def test_the_door_that_pins_the_runs_cliff_pins_its_price(monkeypatch):
    s, dev = _card(monkeypatch, "triton")
    monkeypatch.setenv("NBX_DEFERRED_DRAIN_BYTES", str(512 * MB))
    assert s._deferred_drain_mb(dev) == pytest.approx(512.0)


def _place(s, dev, mem):
    s._get_component_dtype = lambda c, n: "float16"
    s._place_component_fgp = lambda *a, **k: None
    s._spatial_component_tiling = lambda *a, **k: None
    return s._place_component(None, "denoiser", mem, [dev], {}, profile(V100_16GB))


def test_a_component_that_fits_only_without_the_drain_is_not_held_whole_under_triton(monkeypatch):
    s_c, dev_c = _card(monkeypatch, "compiled")
    s_t, dev_t = _card(monkeypatch, "triton")
    cliff = _cliff_mb(dev_t)
    factor = s_t._live_activation_mb(SimpleNamespace(activation_mb=1.0))   # the arena factor, read
    room = s_c._usable_mb(dev_c) - cliff / 2          # what the card holds before the drain is priced
    act = 1024.0
    weight = room - act * factor                       # whole under Triton = room, inside usable - 0
    mem = SimpleNamespace(weight_mb=weight, activation_mb=act, total_mb=weight + act,
                          weight_bytes=int(weight * MB), activation_bytes=int(act * MB),
                          overhead_bytes=0)
    s_t._get_component_dtype = lambda c, n: "float16"
    whole = s_t._whole_component_mb(None, "denoiser", mem, dev_t)
    assert s_c._usable_mb(dev_c) - cliff < whole <= s_c._usable_mb(dev_c), (
        "the scenario no longer sits between the two figures")
    placed = _place(s_t, dev_t, mem)
    assert placed is not None
    assert placed[0] != dev_t.device_string, (
        f"{whole:,.0f} MB held whole on {dev_t.device_string} beside a {cliff:,.0f} MB drain the "
        f"usable {s_t._usable_mb(dev_t) + cliff:,.0f} MB cannot also hold: the plan the allocator "
        f"had to rescue")
    assert placed[0] == f"zero3:{dev_t.device_string}", placed
    # The control: the compiled engine has no queue, and holds the same component whole.
    assert _place(s_c, dev_c, mem)[0] == dev_c.device_string
