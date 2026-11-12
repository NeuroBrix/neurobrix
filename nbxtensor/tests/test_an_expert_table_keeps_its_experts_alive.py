"""The MoE pointer table the certification harness launches through holds bare device addresses: the expert tensors
it was built from must live as long as the table, whatever the caller kept.

Before (the Dell, Volta, 2026-11-12 19:10): `_expert_table([upload(w) for w in ws])` let the uploaded experts die once
the table was built; CUDA then read freed memory and every config of fused_moe_kernel, moe_gateup_vec_half_kernel and
moe_down_split_vec_half_kernel diverged from the fp64 oracle with inf (main, which kept them in a local, passed).
Metal's capture pins hid it.

What this test does if the code is wrong: the experts uploaded inside the call are collected after it — red.

Run: python -m pytest nbxtensor/tests/test_an_expert_table_keeps_its_experts_alive.py
"""
import gc
import sys

import numpy as np


class _AddressesOnly:
    """A table as CUDA's: bare addresses, no pin holding a tensor (Metal's capture pins hold them and hid the bug)."""
    def __init__(self, experts):
        from nbxtensor.memory.allocator import DeviceAllocator
        self.gate_ptrs = {DeviceAllocator.get_device(): [t.data_ptr() for t in experts]}

    def release_pins(self):
        pass


def test_the_experts_live_as_long_as_their_table(monkeypatch):
    """The caller drops its list after the call, as `[upload(w) for w in ws]` does: each expert must still be held
    by something other than this test (the table's keeper), counted by CPython's reference count."""
    from nbxtensor.memory import pointer_tables
    from nbxtensor.ops import autotune_harness as H
    monkeypatch.setattr(pointer_tables, "build_ptr_tables", lambda g, u, d: _AddressesOnly(g))
    rng = np.random.default_rng(0)
    experts = [H.upload(rng.standard_normal((8, 16)).astype(np.float16)) for _ in range(4)]
    alone = [sys.getrefcount(t) for t in experts]        # held by this list only
    H._expert_table(list(experts))
    gc.collect()
    held = [sys.getrefcount(t) for t in experts]
    assert all(h > a for h, a in zip(held, alone)), ("an expert is held by nothing once the call returns while its "
                                                     f"address table is in use: {alone} -> {held}")
