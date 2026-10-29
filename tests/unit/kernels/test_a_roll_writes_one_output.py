"""The Triton engine's `roll` (`kernels/wrappers.roll_wrapper`) equals `torch.roll` and makes ONE output: every rolled
dim's blocks are copied from the input, by its strides, straight into their places. Rolled one dim after the other,
each dim made a whole intermediate: HAT-S at 2048 px on 16 GB, the shifted-window roll on two dims held three
2 304 MB maps where its price holds two (2026-10-22). Needs a CUDA device.

Injection (seen red, then restored green, 2026-10-22): the per-dim `cat` loop back -> the peak cell red."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from nbxtensor.tensor import NBXTensor  # noqa: E402
from nbxtensor.memory.allocator import DeviceAllocator as DA
from neurobrix.kernels.torch_interop import nbx_to_torch  # noqa: E402
from nbxtensor.ops.shape_wrappers import roll_wrapper  # noqa: E402


@pytest.mark.parametrize("shifts,dims", [((-8, -8), (1, 2)), ((3,), (2,)), ((5, -2), (2, 1)), ((0, 4), (1, 2))])
@pytest.mark.parametrize("strided", [False, True])
def test_roll_equals_torchs(shifts, dims, strided):
    a = np.random.default_rng(1).standard_normal((2, 12, 20, 6)).astype(np.float32)
    x = NBXTensor.from_numpy(a).to("cuda")
    t = torch.from_numpy(a)
    if strided:
        x, t = x.permute(0, 2, 1, 3), t.permute(0, 2, 1, 3)
    got = nbx_to_torch(roll_wrapper(x, list(shifts), list(dims))).cpu()
    assert torch.equal(got, torch.roll(t, shifts, dims))


def test_a_two_dim_roll_holds_one_output():
    x = NBXTensor.from_numpy(np.ones((1, 256, 256, 32), dtype=np.float32)).to("cuda")
    roll_wrapper(x, [-8, -8], [1, 2])                            # warm
    dev = DA.get_device()
    DA._pool_flush(dev)
    base = DA._cuda_live_bytes.get(dev, 0)
    DA._cuda_peak_bytes[dev] = base
    out = roll_wrapper(x, [-8, -8], [1, 2])
    peak = DA._cuda_peak_bytes.get(dev, 0) - base
    assert peak <= out.numel() * 4 + (1 << 20), (peak, out.numel() * 4)
