"""A device tensor that dies after interpreter teardown has cleared the allocator's module leaves no error behind.
tensor.py imports the allocator while it imports, so at teardown the allocator's module is cleared first (its globals set
to None): a tensor freed after that raised inside the allocator ("'NoneType' object has no attribute
'_maybe_init_pool'", the GPU kernel suite's exit, 2026-10-29, after the M2 cut of tensor.py). The suite's exact teardown
order does not reproduce in a small child, so the state is set here as teardown leaves it: the allocator module's
`DeviceAllocator` global None. Needs a device."""
import sys

import pytest

import nbxtensor.memory.allocator as allocator_module
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.tensor import NBXTensor


def test_a_tensor_freed_after_the_allocator_module_is_cleared_raises_nothing(monkeypatch):
    if DeviceAllocator.device_count() == 0:
        pytest.skip("no device")
    t = NBXTensor.zeros((1024,), NBXDtype.float32, "cuda:0")
    seen = []
    monkeypatch.setattr(sys, "unraisablehook", lambda u: seen.append(u))
    monkeypatch.setattr(allocator_module, "DeviceAllocator", None)     # as interpreter teardown leaves the module
    del t
    assert not seen, [f"{type(u.exc_value).__name__}: {u.exc_value}" for u in seen]
