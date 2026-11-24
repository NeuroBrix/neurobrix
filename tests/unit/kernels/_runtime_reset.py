"""A test that replaces the process's Metal runtime takes the dead runtime's fault words with it.

`metal_device.reset_runtime_for_tests` drops the runtime and every cache it names as holding it (the launcher's
handles, the allocator's pool and epoch, `gpu_runtime`), but not `faults._FAULT_BUFFERS`: the per-device fault word an
armed kernel binds, allocated once and never freed by contract (a frozen replay plan records its pointer). After a
reset the word is an address of the dropped runtime, and every armed launch after it was refused: "pointer argument 3
... lies in no allocation of this Metal runtime" (`test_no_weight_copies.py::
test_index_select_along_any_axis_needs_no_layout_copy[0,1,2]`, red in the in-order unit run and green alone,
2026-11-24; bisected by scripts/bisect_order_leak.py to the cells of test_metal_allocator.py and
test_metal_runtime_lifetime.py that reset the runtime). The library's reset misses this cache; until it drops it,
a cell that resets runs under `runtime_reset_scoped`.
"""
from __future__ import annotations

import contextlib


@contextlib.contextmanager
def runtime_reset_scoped():
    """Run the block; if it replaced the process's Metal runtime, forget the fault words the dropped runtime owned,
    so the next armed launch allocates its word in the runtime in force."""
    from nbxtensor.backend import faults, metal_device
    before = metal_device._RUNTIME
    try:
        yield
    finally:
        if metal_device._RUNTIME is not before:
            with faults._FAULT_LOCK:
                faults._FAULT_BUFFERS.clear()
