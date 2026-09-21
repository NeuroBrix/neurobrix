"""The in-place-unary interceptor dispatches by tensor KIND before size.

real-esrgan-x8 compiled died at `aten.leaky_relu::0`: `'Tensor' object has
no attribute '_device_idx'`. The interceptor's size test came FIRST, so a
torch tensor below the 1 GB threshold fell to `return _fn(x, ...)` and handed
the NBX/triton wrapper (`wrappers.leaky_relu`) a torch tensor, whose
`NBXTensor.empty_like` reads `_device_idx`. The op-level tiling engine runs
in compiled and sequential modes, where tensors are torch — so a small
activation on the tiled path crashed.

Kind is the outer decision now: a torch tensor takes torch.nn.functional,
never the NBX wrapper; in-place (the shared-buffer optimisation) is applied
only when the tensor is big enough to be worth it and contiguous.

Model-free: the interceptor is built directly and driven with each kind at
each side of the threshold.
"""
from __future__ import annotations

import pytest

from neurobrix.core.module.tiling_engine import OpLevelTilingEngine

torch = pytest.importorskip("torch")

GB = 1024 * 1024 * 1024


def _interceptor(min_bytes=GB):
    return OpLevelTilingEngine.build_inplace_unary_interceptor(
        "aten::leaky_relu", min_bytes)


def test_a_small_torch_tensor_never_reaches_the_nbx_wrapper():
    """The exact crash: torch tensor, below threshold. Before the fix this
    called wrappers.leaky_relu and raised on the missing `_device_idx`."""
    fn = _interceptor()
    x = torch.randn(1, 8, 16, 16)                 # well under 1 GB
    out = fn(x, 0.01)
    assert torch.is_tensor(out)
    # value is the real leaky_relu, and the input is NOT mutated (out-of-place
    # below threshold, so the buffer stays available for other readers)
    import torch.nn.functional as F
    assert torch.allclose(out, F.leaky_relu(x, 0.01))


def test_a_big_torch_tensor_rides_in_place():
    """Above the threshold and contiguous → in-place, same values."""
    fn = _interceptor(min_bytes=1)               # force the big branch
    x = torch.randn(64, 64).contiguous()
    ref = torch.nn.functional.leaky_relu(x.clone(), 0.01)
    out = fn(x, 0.01)
    assert out.data_ptr() == x.data_ptr(), "big contiguous torch not in place"
    assert torch.allclose(out, ref)


def test_a_noncontiguous_big_torch_tensor_falls_out_of_place():
    """A strided view must not be written in place (it aliases its base)."""
    fn = _interceptor(min_bytes=1)
    base = torch.randn(64, 128)
    x = base[:, ::2]                              # non-contiguous view
    assert not x.is_contiguous()
    out = fn(x, 0.01)
    assert out.data_ptr() != x.data_ptr(), "in-place write into a strided view"


def test_the_nbx_path_is_unchanged_below_and_above():
    """The NBX branch: below → out-of-place, above+contiguous → out=x."""
    calls = []

    class _FakeNBX:
        # stands in for NBXTensor via isinstance? No — the interceptor checks
        # isinstance(x, NBXTensor). Use the real type with a tiny tensor.
        pass

    from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator
    if DeviceAllocator.device_count() <= 0:
        pytest.skip("the NBX branch needs a device the wrapper can launch on")
    import numpy as np
    x = NBXTensor.from_numpy(np.random.randn(8, 16).astype(np.float32))
    fn_small = _interceptor(min_bytes=GB)
    out = fn_small(x, 0.01)                       # below threshold: out-of-place
    assert out.data_ptr() != x.data_ptr()
    fn_big = _interceptor(min_bytes=1)
    x2 = NBXTensor.from_numpy(np.random.randn(8, 16).astype(np.float32))
    out2 = fn_big(x2, 0.01)                       # above + contiguous: in place
    assert out2.data_ptr() == x2.data_ptr()
