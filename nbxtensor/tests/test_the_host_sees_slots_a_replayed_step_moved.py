"""The expert store's host residency takes over from the card's only after reading whether the CARD moved its slots —
by the card's own clock (`ExpertSlotStore._card_moved_slots`), never by a flag the host's code sets. The clock is read
at the size of the word it is read into (`nbytes`), no byte count written in the code.

Before (2026-11-12): `_host_takes_over` reset the host's map only when `_host_dirty` said so, and only a DIRECT decode
step sets it (`decode_on_card`). A replayed decode step runs no Python: its slot assignments refilled slots with other
experts unseen, the next prompt's MoE layers (the host's path) read them as the experts its map named, and wrote that
stale map back to the card. A served mixture-of-experts coding model answered the same turn right twice, then "`` ```"; with the
replay off, eleven requests were right.

What this test does if the code is wrong (a store whose card residency is armed, its clock a real device word on
card 0; the residency stood in for, counting resets):
1. the card's clock advanced on the card (what a replayed step's `expert_slot_assign_kernel` does) and no Python flag
   set: the host does not reset its map — red on main;
2. the clock unchanged since the host wrote the map: the host resets it — red (every chunk of one prefill would copy
   its experts again);
3. a direct step's flag still resets it, and the card's map is marked invalid after a take-over — red.

Device test: one card nobody else uses. Run: CUDA_VISIBLE_DEVICES=<free card> python -m pytest <this file>
"""
import numpy as np
import pytest


def _store():
    from nbxtensor.memory.expert_slots import ExpertSlotStore
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.memory.allocator import DeviceAllocator as DA
    try:
        DA.set_device(0)
        clock = NBXTensor.from_numpy(np.array([7], dtype=np.int64), device=0)
    except Exception as e:                              # no card here
        pytest.skip(f"no card: {e}")
    s = object.__new__(ExpertSlotStore)
    s.device_idx, s._clock, s._clock_seen = 0, clock, 7
    s._dev_valid, s._host_dirty = True, False

    class Residency:
        resets = 0

        def reset(self):
            Residency.resets += 1
    s.residency = Residency()
    return s, Residency


def _tick_on_the_card(store):
    from nbxtensor.memory.allocator import DeviceAllocator as DA
    one = np.array([8], dtype=np.int64)
    DA.set_device(0)
    DA.memcpy(store._clock.data_ptr(), one.ctypes.data, 8, 1)


def test_a_clock_moved_by_the_card_resets_the_hosts_map():
    store, R = _store()
    _tick_on_the_card(store)
    store._host_takes_over()
    assert R.resets == 1 and store._dev_valid is False


def test_an_unmoved_clock_keeps_the_hosts_map():
    store, R = _store()
    store._host_takes_over()
    assert R.resets == 0 and store._dev_valid is False


def test_a_direct_steps_flag_still_resets():
    store, R = _store()
    store._host_dirty = True
    store._host_takes_over()
    assert R.resets == 1 and store._host_dirty is False
