"""A compiled sequence's slot memory lives through the library's lifecycle (`nbxtensor.memory.slot_store`): a request's
end keeps the weights and drops the inputs and intermediates; a release synchronises every device holding one of its
tensors and then drops every slot.

Before (2026-11-13): the engine's executor chose which regions to clear and reached into each sequence's arena for it
(`seq._arena.clear_intermediates()`, `.clear_inputs()`, `.release()`): the slot memory's lifecycle was the engine's.

What this test does if the code is wrong: a request's end drops a weight or keeps an input, a release drops the slots
before the devices are synchronised, or the figures read another region — red.

Run: python -m pytest nbxtensor/tests/test_a_slot_store_ends_its_request_and_is_released_by_the_library.py
"""


class Held:
    def __init__(self, nbytes, dev=0, ptr=1):
        self._nbytes, self._device_idx, self._ptr = nbytes, dev, ptr

    def nbytes(self):
        return self._nbytes

    def data_ptr(self):
        return self._ptr


def _filled():
    from nbxtensor import memory
    store = memory.slot_store(5, 2, 1)         # [w, w | in | mid, mid]
    for i, n in enumerate((10, 20, 30, 40, 50)):
        store[i] = Held(n, ptr=i + 1)
    return memory, store


def test_a_requests_end_keeps_the_weights_only():
    memory, store = _filled()
    memory.end_store_request(store)
    assert [None if t is None else t.nbytes() for t in store.slots] == [10, 20, None, None, None]


def test_a_release_synchronises_the_devices_before_it_drops_the_slots(monkeypatch):
    from nbxtensor.memory import deferred_frees
    memory, store = _filled()
    seen = []

    def synced(tensors):
        seen.append(sum(1 for _ in tensors))
        assert all(t is not None for t in store.slots)      # nothing dropped yet
    monkeypatch.setattr(deferred_frees, "sync_devices_of", synced)
    memory.release_store(store)
    assert seen == [5] and all(t is None for t in store.slots)


def test_the_figures_read_the_store_by_region():
    memory, store = _filled()
    assert memory.store_held_bytes(store, 0) == (30, 120)
    c = memory.store_contents(store, 45)
    assert c["weights"] == [2, 30] and c["inputs"] == [1, 30] and c["intermediates"] == [2, 90]
    assert [i for i, _ in c["large"]] == [4]
    assert memory.SlotStore is type(store)
