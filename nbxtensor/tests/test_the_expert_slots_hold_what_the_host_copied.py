"""An expert slot store moves the bytes its caller's residency decides: a routed expert that missed is copied into the
slot it was given, an unrouted entry points at a routed slot, a copy made ahead is waited for by its call, the card's
own map says what the host put in the slots — and the experts of whole layers held on another device are there, bit
for bit, in one arena.

The residency here is a stand-in that answers the store's protocol (`memory.expert_slots`): the library needs nothing
of an engine to hold experts.

What this test would do if the code were wrong: a miss copied into another slot, or from another expert — RED on the
slot's bytes; a slot rewritten by the second call still read as its first expert — RED; a prefetch's event left behind
— RED; the card's map written at another size than its arrays, or from another order than the residency's — RED on the
map read back; a held expert's projection off by one aligned step — RED on its bytes.

Device test: one card nobody else uses. Run: CUDA_VISIBLE_DEVICES=<free card> python -m pytest <this file>
"""
import ctypes
from collections import OrderedDict

import numpy as np
import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.memory.allocator import DeviceAllocator as DA
from nbxtensor.tensor import NBXTensor

E, H, F, SLOTS = 6, 48, 24, 3


class _Residency:
    """Least recently used over `n_slots`, the protocol the store asks and nothing more."""

    def __init__(self, n_slots):
        self.n_slots, self.hits, self.misses, self.resets = n_slots, 0, 0, 0
        self.clear()

    def clear(self):
        self._map, self._free = OrderedDict(), list(range(self.n_slots - 1, -1, -1))

    def reset(self):
        self.resets += 1
        self.clear()

    def _place(self, layer, experts, keep):
        slots, missed = {}, []
        for e in (e for e in experts if (layer, e) in self._map):
            self._map.move_to_end((layer, e))
            slots[e] = self._map[(layer, e)]
        for e in (e for e in experts if e not in slots):
            slot = self._free.pop() if self._free else self._map.pop(next(k for k in self._map if k not in keep))
            self._map[(layer, e)] = slots[e] = slot
            missed.append(e)
        return slots, missed

    def assign(self, layer, routed):
        experts = sorted({int(e) for e in routed})
        return self._place(layer, experts, {(layer, e) for e in experts})

    def reserve(self, layer, experts, protect):
        experts = sorted({int(e) for e in experts})
        slots, missed = self._place(layer, experts, set(protect) | {(layer, e) for e in experts})
        return {e: slots[e] for e in missed}

    def resident(self):
        return list(self._map.items())

    def summary(self):
        return "a stand-in"


def _host(a):
    t = NBXTensor.empty_cpu(a.shape, NBXDtype.float16, pinned=True)
    ctypes.memmove(t.data_ptr(), a.ctypes.data, a.nbytes)
    return t


@pytest.fixture
def experts():
    try:
        DA.set_device(0)
        NBXTensor.zeros((1,), dtype=NBXDtype.float16, device="cuda:0")
    except Exception as e:                                  # no accelerator: a card-less machine
        pytest.skip(f"no accelerator: {e}")
    rng = np.random.default_rng(3)
    values = [tuple(rng.standard_normal(s).astype(np.float16) for s in ((F, H), (F, H), (H, F))) for _ in range(E)]
    return values, [tuple(_host(a) for a in trip) for trip in values]


def _store(host):
    from nbxtensor.memory.expert_slots import ExpertSlotStore
    return ExpertSlotStore(_Residency(SLOTS), 0, host[0])


def _same(device_triple, values):
    DA.sync_device()
    return all(np.array_equal(np.asarray(t.numpy()), a) for t, a in zip(device_triple, values))


def test_a_miss_is_copied_into_its_slot_and_the_rest_point_at_a_routed_one(experts):
    values, host = experts
    store = _store(host)
    try:
        g, u, d = store.acquire("L0", [4, 1], E, lambda e: host[e])
        for e in (1, 4):
            assert _same((g[e], u[e], d[e]), values[e]), f"expert {e}'s slot does not hold expert {e}"
        assert all(g[e] is g[1] for e in (0, 2, 3, 5)), "an unrouted entry points at the lowest routed expert's slot"
        assert store.bytes_in == 2 * sum(a.nbytes for a in values[0])
        assert store.device_bytes >= SLOTS * store.slot_bytes
    finally:
        store.release()


def test_a_slot_given_to_another_expert_holds_the_other_expert(experts):
    values, host = experts
    store = _store(host)
    try:
        store.acquire("L0", [0, 1, 2], E, lambda e: host[e])
        g, u, d = store.acquire("L0", [5, 2], E, lambda e: host[e])        # 5 takes the least recent slot: expert 0's
        assert _same((g[5], u[5], d[5]), values[5]) and _same((g[2], u[2], d[2]), values[2])
        assert g[5].data_ptr() == store.slots[0][0].data_ptr()
        assert store.bytes_in == 4 * sum(a.nbytes for a in values[0]), "a hit crosses the link once"
    finally:
        store.release()


def test_a_copy_made_ahead_is_waited_for_by_its_call(experts):
    values, host = experts
    store = _store(host)
    try:
        store.acquire("L0", [0], E, lambda e: host[e])
        assert store.prefetch("L1", [3, 4], lambda e: host[e], {("L0", 0)}) == 2
        assert set(store._inflight) == {("L1", 3), ("L1", 4)}
        g, u, d = store.acquire("L1", [3, 4], E, lambda e: host[e])
        assert not store._inflight, "the call waits for its prefetched experts and ends their events"
        assert _same((g[3], u[3], d[3]), values[3]) and _same((g[4], u[4], d[4]), values[4])
        assert store.bytes_in == 3 * sum(a.nbytes for a in values[0]), "a prefetched expert is not copied again"
    finally:
        store.release()


def test_a_host_expert_of_another_geometry_is_refused_by_name(experts):
    _values, host = experts
    store = _store(host)
    try:
        wrong = (host[0][0], host[0][2], host[0][2])
        with pytest.raises(RuntimeError, match=r"expert 2 of 'L0' projection 1 .* one geometry per component"):
            store.acquire("L0", [2], E, lambda e: wrong)
    finally:
        store.release()


def _read(t, dtype):
    out = np.zeros(t.numel(), dtype=dtype)
    DA.sync_device()
    DA.memcpy(out.ctypes.data, t.data_ptr(), out.nbytes, 2)
    return out


def test_the_cards_map_says_what_the_host_put_in_the_slots(experts):
    _values, host = experts
    store = _store(host)
    try:
        store.enable_device_decode({"L0": 0, "L1": 1}, E)
        store.acquire("L0", [2, 5], E, lambda e: host[e])
        store.acquire("L1", [1], E, lambda e: host[e])
        slot_of, owner = _read(store._slot_of, np.int32), _read(store._owner, np.int32)
        held = {(layer, e): slot for (layer, e), slot in store.residency.resident()}
        assert held == {("L0", 2): 0, ("L0", 5): 1, ("L1", 1): 2}
        expect = np.full(2 * E, -1, dtype=np.int32)
        expect[[2, 5, E + 1]] = [0, 1, 2]
        assert slot_of.tolist() == expect.tolist() and owner.tolist() == [2, 5, E + 1]
        assert _read(store._stamp, np.int64).tolist() == [0, 1, 2] and _read(store._clock, np.int64).tolist() == [3]
        assert store._clock_seen == 3 and store._dev_valid
        assert store._card_moved_slots() is False
        four = np.array([4], dtype=np.int64)                         # what the card's assign kernel does to its clock
        DA.memcpy(store._clock.data_ptr(), four.ctypes.data, four.nbytes, 1)
        assert store._card_moved_slots() is True
        store.acquire("L0", [2], E, lambda e: host[e])               # the host takes over: its map starts empty
        assert store.residency.resets == 1 and store.residency.resident() == [(("L0", 2), 0)]
    finally:
        store.release()


def test_whole_layers_held_on_a_device_are_there_bit_for_bit(experts):
    from nbxtensor.memory.expert_slots import HeldExperts
    from nbxtensor.memory.layout import aligned
    values, host = experts
    held = HeldExperts(0, {"A": host[:2], "B": host[2:5]})
    try:
        assert held.bytes == sum(aligned(a.nbytes) for trip in values[:5] for a in trip)
        for key, first in (("A", 0), ("B", 2)):
            g, u, d = held.layers[key]
            for i in range(len(g)):
                assert _same((g[i], u[i], d[i]), values[first + i]), f"{key}[{i}]"
        ptrs = sorted(t.data_ptr() for lists in held.layers.values() for lst in lists for t in lst)
        assert all(p % 256 == 0 for p in ptrs) and ptrs[-1] - ptrs[0] < held.bytes, "one arena, aligned views"
    finally:
        held.release()


def test_a_held_expert_that_is_not_a_host_tensor_is_refused_by_name(experts):
    from nbxtensor.memory.expert_slots import HeldExperts
    _values, host = experts
    on_device = NBXTensor.zeros((F, H), dtype=NBXDtype.float16, device="cuda:0")
    with pytest.raises(RuntimeError, match=r"an expert of 'A' held on card 0 .* contiguous"):
        HeldExperts(0, {"A": [(host[0][0], on_device, host[0][2])]})
