"""`NBXTensor.unflatten` splits one dim as a VIEW on any strides (torch's `unflatten`), and a sliced stretch's
`take_slice` therefore copies its SLICE of a strided input, never the whole input. HAT-S at 2048 px on 16 GB
(2026-10-22): the stretch read q, k, v as views of the 6 912 MB qkv, and the `reshape` the slicing did made each whole
input contiguous first — 3 x 2 304 MB beside the qkv on every pass, unpriced. Needs a CUDA device.

Injection (seen red, then restored green, 2026-10-22): `take_slice` back to `x.reshape(...)` -> the peak cell red."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from nbxtensor.tensor import NBXTensor  # noqa: E402
from nbxtensor.memory.allocator import DeviceAllocator as DA
from neurobrix.kernels.torch_interop import nbx_to_torch  # noqa: E402
from neurobrix.core.prism import chunked_region as CR  # noqa: E402


def _strided():
    a = np.arange(4 * 6 * 24 * 8, dtype=np.float32).reshape(4, 6 * 24, 8)
    x = NBXTensor.from_numpy(a).to("cuda").transpose(1, 2)        # (4, 8, 144), non-contiguous
    return a, x


def test_unflatten_is_a_view_equal_to_torchs():
    a, x = _strided()
    u = x.unflatten(2, (6, 24))
    ref = torch.from_numpy(a).transpose(1, 2).unflatten(2, (6, 24))
    assert tuple(u.shape) == tuple(ref.shape)
    assert u.data_ptr() == x.data_ptr()                          # no copy
    assert np.array_equal(nbx_to_torch(u.contiguous()).cpu().numpy(), ref.contiguous().numpy())


def test_a_slice_of_a_strided_input_copies_the_slice_only():
    big = NBXTensor.from_numpy(np.random.default_rng(0).standard_normal((64, 512, 96)).astype(np.float32)).to("cuda")
    x = big.transpose(1, 2)                                      # (64, 96, 512): the token axis 2, strided
    lay = SimpleNamespace(axis=2)                                # take_slice reads the layout's axis
    total, inner, start, length = 512, 1, 128, 64
    CR.take_slice(x, lay, total, inner, start, length)           # warm
    dev = DA.get_device()
    DA._pool_flush(dev)
    base = DA._cuda_live_bytes.get(dev, 0)
    DA._cuda_peak_bytes[dev] = base
    got = CR.take_slice(x, lay, total, inner, start, length)
    peak = DA._cuda_peak_bytes.get(dev, 0) - base
    slice_bytes = 64 * 96 * 64 * 4
    ref = nbx_to_torch(x.contiguous()).cpu()[:, :, start:start + length]
    assert np.array_equal(nbx_to_torch(got).cpu().numpy(), ref.numpy())
    assert peak <= 2 * slice_bytes, (peak, slice_bytes, 64 * 96 * 512 * 4)
