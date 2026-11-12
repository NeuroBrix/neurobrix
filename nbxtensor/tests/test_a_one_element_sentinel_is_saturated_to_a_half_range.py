"""`NBXTensor.saturated(dtype)`: a one-element float32 / float64 tensor holding a finite value beyond a half dtype's
range comes back as a new one-element tensor holding the clamped value; anything else is returned as itself. The
engine's `where` wrap calls it on its operands (a mask sentinel written for bfloat16 kept finite in float16). It lived
in the engine (triton/dtype.py `_saturated_one_element`) until 2026-11-12.

What this test would do if the code were wrong: -3.39e38 kept (a -inf after the cast to float16), a value already in
range or an infinity rewritten, a tensor of more than one element or of a half dtype read, the dtype or shape of the
result changed — RED."""
import numpy as np
import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.tensor import NBXTensor


def _accelerator():
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() == 0:          # only a machine without an accelerator skips; any other error fails
        pytest.skip("no accelerator on this machine")


def test_a_sentinel_beyond_float16_is_clamped_and_keeps_its_dtype_and_shape():
    _accelerator()
    t = NBXTensor.from_numpy(np.full((1, 1), -3.39e38, np.float32))
    s = t.saturated(NBXDtype.float16)
    assert s is not t and s.nbx_dtype == NBXDtype.float32 and s.shape == (1, 1)
    assert s.numpy().tolist() == [[-65504.0]]
    inside = NBXTensor.from_numpy(np.full((1, 1), -1e38, np.float32))
    assert inside.saturated("bfloat16") is inside                         # within bfloat16's range: unchanged


def test_what_needs_no_clamp_is_returned_as_itself():
    _accelerator()
    for t in (NBXTensor.from_numpy(np.array([1.5], np.float32)),
              NBXTensor.from_numpy(np.array([-np.inf], np.float32)),
              NBXTensor.from_numpy(np.array([7e4, 8e4], np.float32)),          # more than one element
              NBXTensor.from_numpy(np.array([7e4], np.float32)).to(NBXDtype.bfloat16)):   # a half tensor itself
        assert t.saturated(NBXDtype.float16) is t
    assert NBXTensor.from_numpy(np.array([7e4], np.float32)).saturated(NBXDtype.float32).numpy().tolist() == [7e4]
