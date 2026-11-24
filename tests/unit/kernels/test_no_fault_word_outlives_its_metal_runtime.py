"""No cell leaves the process an armed kernel's fault word of a dropped Metal runtime: the cell that proves a discrete
device is refused leaves the process's runtime as it found it, and a cell that resets the runtime takes the dead
runtime's fault words with it (the library's own `metal_device.reset_runtime_for_tests`, from nbxtensor 3e51f191).

It used to drop the runtime (`metal_device.reset_runtime_for_tests`) around the refused construction. Before
nbxtensor 3e51f191 the reset kept the per-device fault word an armed kernel binds, never freed by contract, so the
word kept an address of the dropped runtime: every armed launch after it was refused, "pointer argument 3 ... lies in
no allocation of this Metal runtime" (`test_no_weight_copies.py::test_index_select_along_any_axis_needs_no_layout_copy
[0,1,2]`, red in the in-order unit run and green alone, 2026-11-24; bisected by scripts/bisect_order_leak.py to
`test_a_device_without_unified_memory_is_refused`).

The lifetime cells of test_metal_runtime_lifetime.py reset the runtime on purpose (the reset is what they prove); the
library's reset now drops the fault words and the device assert's status blocks itself, so they run bare.

What this test would do if the code were wrong: with the reset back in the refusal cell, the runtime after it is
another object — red; with a library whose reset keeps the fault words (50597afc), the word after a lifetime cell is
the dropped runtime's — red. Both are asked through public calls only."""
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


@pytest.mark.skipif(not _on_metal(), reason="needs an Apple GPU")
def test_a_cell_that_resets_the_runtime_leaves_a_fault_word_of_the_runtime_in_force():
    """Arm the fault word, run a lifetime cell that resets the runtime through the library's own reset, and find the
    word an armed launch binds next in an allocation of the runtime in force. With a reset that keeps the word, it is
    the dropped runtime's — red."""
    from nbxtensor.backend import faults, metal_device
    from nbxtensor.memory.allocator import DeviceAllocator
    import tests.unit.kernels.test_metal_runtime_lifetime as lifetime
    device = DeviceAllocator.get_device()
    faults.device_fault_buffer(device)
    lifetime.test_resetting_the_runtime_clears_every_cache_that_holds_it()
    word = faults.device_fault_buffer(device)
    buffer, _offset = metal_device.runtime().buffer_for_pointer(int(word.data_ptr()))
    assert buffer is not None, "the fault word an armed launch binds lies in no allocation of the runtime in force"
