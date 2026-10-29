"""The library's device files (backend/devices): a device is described by the library alone, and an application's
half of the same device merges with it, a key written in both refused (API.md section 11)."""
import pytest

from nbxtensor.backend import device_files as D


def test_a_device_is_described_by_the_library_alone():
    m4 = D.read("apple", "apple_m4_pro")
    assert m4["compute_capability"] == "apple-m4-pro"                 # identification: the device is found by it
    assert m4["memory"]["max_shared_memory_per_block"] > 0            # what the kernels are tiled by
    assert "measured" not in m4                                       # an application's rates are not the library's


def test_an_undescribed_device_has_no_library_half():
    assert D.read("apple", "no_such_chip") == {}


def test_the_two_halves_merge_and_a_fact_is_written_once():
    lib = {"memory": {"warp_size": 32}, "flash": {"x": 1}}
    app = {"memory": {"budget_mb": 100}, "measured": {"gbps": 3}}
    assert D.merge(lib, app) == {"memory": {"warp_size": 32, "budget_mb": 100}, "flash": {"x": 1},
                                 "measured": {"gbps": 3}}
    with pytest.raises(ValueError, match="memory.warp_size: written in both"):
        D.merge(lib, {"memory": {"warp_size": 64}})
