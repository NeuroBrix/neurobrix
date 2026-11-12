"""The certifier gives the device back between keys through the library's release, called by name.

Red on 2026-09-22: `_release_between_keys` probed `empty_cache` then `device_empty_cache`.
DeviceAllocator has neither — its pool drain is `empty_cache_pool` — so the drain resolved to
None and the pool was never drained; a whole Apple certification campaign ran with it inert and
reached a 21.9 GB footprint on a 24 GB machine. Since 2026-11-12 the hook calls
`nbxtensor.memory.memory_pool.release_flow_memory`, which collects and empties the pool.
"""
from nbxtensor.memory import memory_pool
from nbxtensor.memory.allocator import DeviceAllocator
from neurobrix.kernels import autotune_certify as AC


def test_release_between_keys_empties_the_allocator_s_pool(monkeypatch):
    calls = []
    monkeypatch.setattr(DeviceAllocator, "empty_cache_pool", staticmethod(lambda: calls.append(1) or 0))
    AC._release_between_keys()
    assert calls == [1], "_release_between_keys did not empty the allocator's pool"
    assert memory_pool.release_flow_memory.__module__ == "nbxtensor.memory.memory_pool"


def test_the_per_key_state_reads_the_allocator_s_public_figures():
    line = AC._alloc_state()
    assert line.startswith("live ") and " pool " in line, line
