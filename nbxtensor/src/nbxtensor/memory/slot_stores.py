"""A compiled sequence's slot memory and its lifecycle: the store a sequence takes at compile (`slot_store`), what a
request's end leaves of it (`end_store_request`: the weights; the inputs and intermediates dropped), and its release at an
unload (`release_store`: every device holding one of its tensors synchronised first, then every slot dropped). The
application calls one of these per event and never drives the slots itself (2026-11-13: the engine's executor chose
which regions to clear and reached into each sequence's arena for it)."""
from nbxtensor.memory.arena import Arena as SlotStore

__all__ = ["SlotStore", "slot_store", "end_store_request", "release_store", "store_held_bytes", "store_contents"]


def slot_store(size: int, num_weights: int = 0, num_inputs: int = 0) -> SlotStore:
    """A store of `size` slots laid out [weights | inputs | intermediates]."""
    return SlotStore(size, num_weights, num_inputs)


def end_store_request(store: SlotStore) -> None:
    """A request ended: its inputs and intermediates dropped, the weights kept for the next one."""
    store.clear_intermediates()
    store.clear_inputs()


def release_store(store: SlotStore) -> None:
    """An unload: every device holding one of the store's tensors synchronised (a kernel still queued may read them),
    then every slot dropped, the weights included."""
    store.release()


def store_held_bytes(store: SlotStore, device_idx: int) -> tuple:
    """(weights, the rest): the bytes the store holds on `device_idx`, each allocation counted once."""
    return store.bytes_on(device_idx)


def store_contents(store: SlotStore, large: int) -> dict:
    """What the store holds, by region, and its intermediates of at least `large` bytes (`Arena.census`)."""
    return store.census(large)
