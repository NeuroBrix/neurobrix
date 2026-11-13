"""The certifier records the target it proves on, the card it ran on and the allocator's figures per key through the
library's public doors (`nbxtensor.device`, `nbxtensor.memory`), never through `DeviceAllocator` or the launcher.

What this test does if the code is wrong: the certifier reads another name than the door (a patched door is not
seen), or its per-key line drops a figure — red.

Run: python -m pytest tests/unit/kernels/test_a_certification_reads_the_device_and_the_allocator_through_the_library_doors.py
"""
from types import SimpleNamespace


def test_the_machine_s_target_is_the_device_door_s(monkeypatch):
    from nbxtensor import device
    from neurobrix.kernels import autotune_certify as AC
    monkeypatch.setattr(device, "target_name", lambda: "cuda-70")
    monkeypatch.setattr(AC, "_certifying_device", lambda: None)
    monkeypatch.setattr(AC, "_clocks_mhz", lambda: None)
    assert AC._machine()["target"] == "cuda-70"


def test_the_certifying_card_is_the_device_door_s_current(monkeypatch):
    from nbxtensor import device
    from nbxtensor.ops import wrappers as W
    from neurobrix.kernels import autotune_certify as AC
    monkeypatch.setattr(device, "current", lambda: 1)
    cards = [SimpleNamespace(index=i, name=f"card{i}", memory_gb=16 * (i + 1)) for i in (0, 1)]
    monkeypatch.setattr(W, "get_hardware_profile", lambda: SimpleNamespace(devices=cards))
    got = AC._read_certifying_device()
    assert got is not None and got["ordinal"] == 1


def test_the_per_key_line_reads_the_memory_door(monkeypatch):
    from nbxtensor import memory
    from neurobrix.kernels import autotune_certify as AC
    monkeypatch.setattr(memory, "memory_allocated", lambda: 3 * 2**20)
    monkeypatch.setattr(memory, "pool_cached_bytes", lambda: 2 * 2**20)
    monkeypatch.setattr(memory, "pool_enabled", lambda: False)
    monkeypatch.setattr(memory, "pool_stats", lambda: {"flushes": 4, "evictions": 5})
    assert AC._alloc_state() == "live 3MB, pool 2MB, pool OFF, flush 4/evict 5"
