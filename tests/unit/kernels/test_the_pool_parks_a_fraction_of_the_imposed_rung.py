"""The Triton allocator's pool parks freed blocks up to `memory.alloc_pool_parked_cap_fraction` of the device; under an
imposed rung (`NBX_PRISM_BUDGET_MB`: the device the plan was sized for) the fraction is of THAT rung. Measured
2026-10-27: mochi-1-preview triton at the 8 GB rung on a 32 GB card parked its 9 GB text encoder arena under a 16 GB
cap and peaked at 22.6 GB; with the cap of the rung, 11.1 GB, output byte-identical. What the test would do if the code
were wrong: the cap of the physical card under the door (red)."""
import pytest

from neurobrix.kernels import nbx_tensor as N
from neurobrix.kernels import wrappers as W


@pytest.fixture
def card(monkeypatch):
    monkeypatch.setattr(N.DeviceAllocator, "_pool_cap_bytes", {})
    monkeypatch.setattr(N.DeviceAllocator, "device_total_bytes", staticmethod(lambda dev: 32 << 30))
    monkeypatch.setattr(W, "get_hardware_profile", lambda: type("P", (), {"devices": [object()]})())
    monkeypatch.setattr(W, "_arch_memory_param", lambda key, default: 0.5)


def test_the_cap_is_a_fraction_of_the_card(card, monkeypatch):
    monkeypatch.delenv("NBX_PRISM_BUDGET_MB", raising=False)
    assert N.DeviceAllocator._pool_parked_cap(0) == 16 << 30


def test_under_an_imposed_rung_the_cap_is_a_fraction_of_the_rung(card, monkeypatch):
    monkeypatch.setenv("NBX_PRISM_BUDGET_MB", "8192")
    assert N.DeviceAllocator._pool_parked_cap(0) == 4 << 30
