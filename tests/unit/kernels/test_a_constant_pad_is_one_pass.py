"""The Triton engine's constant_pad_nd pads any dims in ONE pass: the output buffer alone, nothing else allocated while
it runs. Its former general N-D path padded one dim at a time through `cat`, holding one output-sized copy per padded dim
beside an unused output buffer: Wan2.1's causal pad [1,1,1,1,2,0] on a (1,96,14,480,832) fp32 tensor held 6,182 MB beside
a 2,060 MB result (measured 2026-10-17), unpriced by any plan. Oracle: torch.nn.functional.pad (the compiled engine's op).

What it would do if the code were wrong: a wrong index map gives a different tensor than torch (red); extra buffers
make the in-op allocation exceed the output's bytes (red)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")                       # the oracle only: torch.nn.functional.pad

from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator as DA  # noqa: E402
from neurobrix.kernels import wrappers as W  # noqa: E402

# the engine's own device gate (CUDA, ROCm or Metal — the Triton tests' gate), never torch.cuda's: gated on CUDA the
# test skipped on the Mac, and a test that never runs on a backend proves nothing for it (the Mac, 2026-10-17 23:04)
if DA.device_count() <= 0:
    pytest.skip("the Triton pad runs on a device", allow_module_level=True)

CASES = [
    ((1, 4, 6, 8, 10), [1, 1, 1, 1, 2, 0], 0.0),      # a causal 5-D pad: W, H and the middle time dim
    ((1, 4, 6, 8, 10), [0, 0, 0, 0, 3, 1], 0.0),      # time only
    ((2, 3, 5, 7), [1, 2, 3, 0], 1.5),                # 4-D, the 2-D kernel's case, a value
    ((3, 5, 7), [0, 0, 2, 1], 0.0),                   # 3-D middle dim: the general path
    ((5, 7), [2, 3], -1.0),                           # 1-D kernel
]


@pytest.mark.parametrize("shape,pad,value", CASES, ids=[f"{s}:{p}" for s, p, _ in CASES])
def test_the_pad_is_torch_s_and_allocates_its_output_only(shape, pad, value):
    x = torch.randn(*shape, dtype=torch.float32)
    want = torch.nn.functional.pad(x, pad, value=value)
    nx = NBXTensor.from_numpy(x.numpy()).to_cuda()
    DA.empty_cache_pool()
    DA.reset_peak_memory(0)
    before = DA.memory_allocated(0)
    got = W.constant_pad_nd_wrapper(nx, pad, value)
    extra = DA.peak_memory_allocated(0) - before
    out_bytes = int(got.numel()) * int(got.element_size())
    assert extra <= out_bytes, f"{extra} B allocated while padding, the output is {out_bytes} B"
    assert tuple(got.shape) == tuple(want.shape)
    np.testing.assert_array_equal(got.numpy(), want.numpy())
