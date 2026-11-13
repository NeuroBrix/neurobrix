"""nbxtensor.memory.step_stabilizer: runs 1-2 pass through, run 3 opens the measuring window, the measurement is banked
into the sequence that opened it at the next tick of ANY sequence, and runs 4+ serve one recycled slab cut from it.

What this test does if the code is wrong: the lifecycle cell fails when a slab is in force before run 4, when it is cut
from anything but the measured bytes, or when it is cut again instead of recycled; the attribution cell fails when
another sequence's tick banks the window into itself (the slab once sized from the sampler's tiny window and starved
by 627 MB); the step-slab cell fails when a real StepSlab cannot carve (its carve record never set: every
stabilized run failed with "'StepSlab' object has no attribute 'carve_sizes'", the Mac 2026-11-13, main too).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from nbxtensor.memory import step_stabilizer as SS


class _Slab:
    cut = []

    def __init__(self, nbytes, dev_idx):
        self.size, self.dev_idx, self.resets = nbytes, dev_idx, 0
        self.allocs, self.stale, self.shortfall, self.prev_carve, self.carve_sizes = {}, set(), 0, None, []

    @classmethod
    def for_carved(cls, carved, dev_idx):
        cls.cut.append((carved, dev_idx))
        return cls(carved, dev_idx)

    def reset(self):
        self.resets += 1


def _state():
    return SimpleNamespace(stab_slab="before", measuring=False, live=7, high_water=7, measured={1: 2},
                           total_measured=9, measure_owner=None)


@pytest.fixture(autouse=True)
def _slabs(monkeypatch):
    _Slab.cut = []
    monkeypatch.setattr(SS, "StepSlab", _Slab)


def test_two_runs_pass_the_third_measures_then_one_slab_is_recycled():
    state = _state()
    s = SS.StepStabilizer(state, "decoder")
    s.tick(0); s.tick(0)
    assert state.stab_slab is None and not state.measuring
    s.tick(0)                                                    # run 3: the window opens on zeroed counters
    assert state.measuring and state.measure_owner is s and state.stab_slab is None
    assert (state.live, state.high_water, state.measured, state.total_measured) == (0, 0, {}, 0)
    state.total_measured = 4096                                  # what run 3 carved
    s.tick(0)                                                    # run 4: banked, cut, in force
    assert not state.measuring and state.measure_owner is None
    assert _Slab.cut == [(4096, 0)] and state.stab_slab is s.slab and s.slab.resets == 1
    s.tick(0); s.tick(0)
    assert _Slab.cut == [(4096, 0)] and state.stab_slab is s.slab and s.slab.resets == 3


def test_another_sequence_s_tick_banks_the_window_into_its_opener():
    state = _state()
    decoder, sampler = SS.StepStabilizer(state, "decoder"), SS.StepStabilizer(state, "sampler")
    for _ in range(3):
        decoder.tick(0)
    state.total_measured = 1 << 20
    sampler.tick(0)                                              # the sampler's first run: the decoder's bytes banked
    assert decoder.total == 1 << 20 and sampler.total is None and state.stab_slab is None
    decoder.tick(0)
    assert _Slab.cut == [(1 << 20, 0)] and state.stab_slab is decoder.slab


def test_a_real_step_slab_carves_records_its_sizes_and_recycles(monkeypatch):
    from nbxtensor.memory import slab as SL
    monkeypatch.setattr(SL.DeviceAllocator, "malloc_direct", staticmethod(lambda nbytes, dev_idx: 1 << 20))
    slab = SL.StepSlab.for_carved(1000, None)
    try:
        assert slab.size == int(1000 * SL.StepSlab.MARGIN_RATIO) + SL.StepSlab.MARGIN_BYTES
        first = [slab.malloc(100), slab.malloc(600)]
        assert slab.carve_sizes == [100, 600] and slab.prev_carve is None
        slab.reset()
        assert [slab.malloc(100), slab.malloc(600)] == first     # the same addresses at the next step
        assert slab.prev_carve == [100, 600] and slab.free(first[0])
    finally:
        SL.ACTIVE_SLABS.remove(slab)
