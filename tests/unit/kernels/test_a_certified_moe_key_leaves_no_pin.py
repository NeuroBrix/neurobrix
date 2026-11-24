"""A MoE key certified inside a test leaves the driver's pin registry as it found it.

The library's certification harness keeps the last eight keys' expert tables pinned for the process
(`autotune_harness._EXPERT_TABLES`, by design for a certification run). In a pytest process the fused-MoE device cell
left eight expert bases pinned at count 3 in `triton_ext_driver._PIN_COUNTS`, and four pin tests of tests/unit/triton
after it went red in the in-order unit run, green alone (2026-11-24; bisected by scripts/bisect_order_leak.py to
`test_a_fused_moe_key_is_priced_and_windowed.py::test_a_windowed_key_certifies_on_the_device`).

What this test would do if the code were wrong: with `_expert_tables.expert_tables_released` emptied (the tables kept), the registry
after the certification holds bases it did not hold before — red."""
from __future__ import annotations

import pytest

from tests.unit.kernels._expert_tables import expert_tables_released


def test_the_certification_releases_every_pin_it_took(monkeypatch):
    """Certify the device cell's key inside `expert_tables_released` and find the pin registry and the harness's
    tables as they were before."""
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() < 1:
        pytest.skip("needs a device")
    from nbxtensor.backend import triton_ext_driver as D
    from nbxtensor.ops import autotune_harness as H
    import tests.unit.kernels.test_a_fused_moe_key_is_priced_and_windowed as cell
    pins, tables = dict(D._PIN_COUNTS), list(H._EXPERT_TABLES)
    with expert_tables_released():
        cell.certify_a_windowed_key(monkeypatch)
    assert dict(D._PIN_COUNTS) == pins, f"the certification left {len(D._PIN_COUNTS) - len(pins)} base(s) pinned"
    assert len(H._EXPERT_TABLES) == len(tables), "the certification left its expert tables in the harness"
