"""`nbxtensor.memory.peak_memory_allocated` and `reset_peak_memory`: an application reads the allocator's live-bytes
peak and restarts it without naming `DeviceAllocator`. Runs on a CPU (the counters are plain tables).

What this test would do if the code were wrong: the functions missing from `nbxtensor.memory`, a peak read for the
wrong device, a reset that leaves the old peak or touches another device — RED."""
from nbxtensor import memory
from nbxtensor.memory.allocator import DeviceAllocator as DA


def test_the_peak_is_read_per_device_and_summed_and_a_reset_restarts_it_from_the_live_bytes(monkeypatch):
    monkeypatch.setattr(DA, "_cuda_peak_bytes", {0: 900, 1: 500})
    monkeypatch.setattr(DA, "_cuda_live_bytes", {0: 100, 1: 200})
    assert memory.peak_memory_allocated(0) == 900
    assert memory.peak_memory_allocated() == 1400
    memory.reset_peak_memory(0)
    assert memory.peak_memory_allocated(0) == 100 and memory.peak_memory_allocated(1) == 500
    memory.reset_peak_memory()
    assert memory.peak_memory_allocated() == 300
