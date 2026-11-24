"""The cell that proves a discrete Metal device is refused leaves the process's runtime, and every buffer bound to it,
as it found them.

It used to drop the runtime (`metal_device.reset_runtime_for_tests`) around the refused construction. The reset does
not reach `faults._FAULT_BUFFERS`, the per-device fault word an armed kernel binds and that is never freed by
contract, so the word kept an address of the dropped runtime: every armed launch after it was refused, "pointer
argument 3 ... lies in no allocation of this Metal runtime" (`test_no_weight_copies.py::
test_index_select_along_any_axis_needs_no_layout_copy[0,1,2]`, red in the in-order unit run and green alone,
2026-11-24; bisected by scripts/bisect_order_leak.py to `test_a_device_without_unified_memory_is_refused`).

What this test would do if the code were wrong: with the reset back in that cell, the runtime after it is another
object and the fault word lies in none of its allocations — red at either assertion."""
from __future__ import annotations

import pytest

import nbxtensor.backend.runtime as _nbx_runtime


def _on_metal() -> bool:
    """Whether this process's GPU backend is Metal (the only backend the cell under test runs on)."""
    try:
        return _nbx_runtime.gpu_backend() == "metal"
    except Exception:
        return False


@pytest.mark.skipif(not _on_metal(), reason="needs an Apple GPU")
def test_the_refusal_cell_keeps_the_runtime_and_its_fault_word():
    """Arm the fault word, run the refusal cell, and find the same runtime and the word still in one of its
    allocations."""
    from nbxtensor.backend import faults, metal_device
    from nbxtensor.memory.allocator import DeviceAllocator
    import tests.unit.kernels.test_metal_allocator as cells
    word = faults.device_fault_buffer(DeviceAllocator.get_device())
    before = metal_device.runtime()
    cells.test_a_device_without_unified_memory_is_refused()
    after = metal_device.runtime()
    assert after is before, "the refusal cell replaced the process's Metal runtime"
    buffer, _offset = after.buffer_for_pointer(int(word.data_ptr()))
    assert buffer is not None, "the fault word lies in no allocation of the runtime in force"
