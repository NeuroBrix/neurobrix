"""The MoE expert tables a test's certification builds, released when the test is done with them.

`autotune_harness._expert_table` keeps the last eight keys' tables, their pins AND the expert tensors, for the
process (a table holds bare addresses: Volta read freed memory without it, 2026-11-12). A pytest process is not a
certification run: each cell that certified or synthesised a MoE key on the card (the fused-MoE device cell, the
certifier's table-builder cell) left eight expert bases pinned at count 3 (gate, up, down) in
`triton_ext_driver._PIN_COUNTS`, and every pin test of tests/unit/triton after it in the same process found the driver
still pinning them (2026-11-24; bisected by scripts/bisect_order_leak.py).
"""
from __future__ import annotations

import contextlib


@contextlib.contextmanager
def expert_tables_released():
    """Every expert table the library's harness builds inside the block is released at its exit, its pins with it;
    the tables held before the block are left as they were."""
    from nbxtensor.memory.allocator import DeviceAllocator
    from nbxtensor.ops import autotune_harness as H
    held = list(H._EXPERT_TABLES)
    try:
        yield
    finally:
        added = [entry for entry in H._EXPERT_TABLES if not any(entry is h for h in held)]
        if added:
            DeviceAllocator.sync_device()
        for tables, _experts in added:
            tables.release_pins()
        H._EXPERT_TABLES[:] = [entry for entry in H._EXPERT_TABLES if any(entry is h for h in held)]
