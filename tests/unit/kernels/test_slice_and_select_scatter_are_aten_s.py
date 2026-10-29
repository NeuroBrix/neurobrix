"""The Triton engine's slice_scatter / select_scatter are ATen's, element for element.

They are the functional form Forge records for an in-place write into a slice / select view
(`GraphBuilder.write_back_view`): Swin's shifted-window mask `img_mask[:, h, w, :] = cnt` reached the containers as a
fill nothing read — an all-zero mask (2026-10-21). The compiled engine runs ATen's own ops; this holds the Triton
metadata ops (kernels/dispatch.py `_meta_slice_scatter`, `_meta_select_scatter`) to them, at the bounds the trace
records: a negative start, a negative end, Python's open end (the int64 maximum), a step of two, a negative index.

What this test does if the code is wrong: a write at the wrong place, a base mutated instead of copied, or a bound
read another way than ATen's gives other values than torch.ops.aten's: red.
"""
import numpy as np
import pytest
import torch

from nbxtensor.ops.dispatch import _meta_select_scatter, _meta_slice_scatter
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.tensor import NBXTensor

OPEN = 2 ** 63 - 1


def _has_gpu() -> bool:
    try:
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_gpu(), reason="needs a device")


def _nbx(a):
    return NBXTensor.from_numpy(np.ascontiguousarray(a)).to_cuda()


@pytest.mark.parametrize("dim,start,end,step", [
    (1, 0, -4, 1), (1, -4, -2, 1), (1, -2, OPEN, 1), (2, 0, OPEN, 1), (2, 1, 7, 2), (0, 0, OPEN, 1),
])
def test_a_slice_scatter_writes_where_aten_s_does(dim, start, end, step):
    rng = np.random.default_rng(dim * 100 + step)
    base = rng.standard_normal((1, 12, 9, 2)).astype(np.float32)
    view = torch.ops.aten.slice(torch.from_numpy(base), dim, start, end, step)
    src = rng.standard_normal(tuple(view.shape)).astype(np.float32)
    want = torch.ops.aten.slice_scatter(torch.from_numpy(base), torch.from_numpy(src), dim, start, end, step).numpy()
    x = _nbx(base)
    got = _meta_slice_scatter(x, _nbx(src), dim, start, end, step)
    assert tuple(got.shape) == want.shape
    assert np.array_equal(got.contiguous().numpy(), want)
    assert np.array_equal(x.contiguous().numpy(), base), "the base was written in place, not copied"


@pytest.mark.parametrize("dim,index", [(1, 3), (1, -1), (2, 0), (3, -2)])
def test_a_select_scatter_writes_where_aten_s_does(dim, index):
    rng = np.random.default_rng(dim * 10 + index + 50)
    base = rng.standard_normal((1, 12, 9, 2)).astype(np.float32)
    src = rng.standard_normal(tuple(torch.from_numpy(base).select(dim, index).shape)).astype(np.float32)
    want = torch.ops.aten.select_scatter(torch.from_numpy(base), torch.from_numpy(src), dim, index).numpy()
    got = _meta_select_scatter(_nbx(base), _nbx(src), dim, index)
    assert np.array_equal(got.contiguous().numpy(), want)
