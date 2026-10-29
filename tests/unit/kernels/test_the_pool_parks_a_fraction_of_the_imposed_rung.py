"""The Triton allocator's pool parks freed blocks up to `memory.alloc_pool_parked_cap_fraction` of the device; under a
limit (`nbxtensor.memory.set_limit`, which NeuroBrix sets from an imposed rung) the fraction is of THAT limit. Measured
2026-10-27: mochi-1-preview triton at the 8 GB rung on a 32 GB card parked its 9 GB text encoder arena under a 16 GB
cap and peaked at 22.6 GB; with the cap of the rung, 11.1 GB, output byte-identical. What the test would do if the code
were wrong: the cap of the physical card under the door (red)."""
import pytest

from nbxtensor import tensor as N
import nbxtensor.memory.allocator as _nbx_allocator
from nbxtensor.ops import wrappers as W


@pytest.fixture
def card(monkeypatch):
    monkeypatch.setattr(_nbx_allocator.DeviceAllocator, "_pool_cap_bytes", {})
    monkeypatch.setattr(_nbx_allocator.DeviceAllocator, "_limit_bytes", {})
    monkeypatch.setattr(_nbx_allocator.DeviceAllocator, "_rung_limit_bytes", {})
    monkeypatch.setattr(_nbx_allocator.DeviceAllocator, "device_total_bytes", staticmethod(lambda dev: 32 << 30))
    monkeypatch.setattr(W, "get_hardware_profile", lambda: type("P", (), {"devices": [object()]})())
    monkeypatch.setattr(W, "_arch_memory_param", lambda key, default: 0.5)


def test_the_cap_is_a_fraction_of_the_card(card, monkeypatch):
    assert _nbx_allocator.DeviceAllocator._pool_parked_cap(0) == 16 << 30


def test_under_an_imposed_rung_the_cap_is_a_fraction_of_the_rung(card, monkeypatch):
    _nbx_allocator.DeviceAllocator.set_limit(8192 << 20)
    assert _nbx_allocator.DeviceAllocator._pool_parked_cap(0) == 4 << 30
