"""aten::sum and aten::mean carry ATen's `dtype` keyword in the Triton branch: the input cast to it first.

Ming-Lite-Omni-1.5's language model, retraced and installed 2026-10-14: its router normalises the top-k weights with
`sum(topk_weights, -1, keepdim=True, dtype=torch.float32)`; the Triton wrapper took no `dtype` and the run died at
aten.sum::0 ("unexpected keyword argument 'dtype'"), the compiled branch rendered the apple.

What this test does if the code is wrong: the keyword cells fail with that TypeError (red before this change) or when
the result does not carry the requested dtype; the value cells fail when the bf16 input is not summed in the requested
dtype; the refusal cell fails when an unknown dtype string is accepted.

Run: python -m pytest tests/unit/kernels/test_a_reduction_takes_its_dtype.py   (needs a CUDA card)
"""
import os

import numpy as np
import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.tensor import NBXTensor

pytestmark = pytest.mark.skipif(os.environ.get("CUDA_VISIBLE_DEVICES", None) == "", reason="needs a device")


def _bf16(a):
    return NBXTensor.from_numpy(a.astype(np.float32)).to(NBXDtype.bfloat16)


@pytest.mark.parametrize("dtype", [{"type": "dtype", "value": "torch.float32"}, "torch.float32", NBXDtype.float32])
def test_sum_takes_its_dtype(dtype):
    from nbxtensor.ops.reduction_wrappers import sum_wrapper
    a = np.array([[0.25, 0.5, 0.125], [1.0, 2.0, 4.0]], dtype=np.float32)
    out = sum_wrapper(_bf16(a), [-1], True, dtype=dtype)
    assert out.nbx_dtype == NBXDtype.float32 and tuple(out.shape) == (2, 1)
    assert np.allclose(out.numpy().ravel(), a.sum(-1))


def test_mean_takes_its_dtype():
    from nbxtensor.ops.reduction_wrappers import mean_wrapper
    a = np.array([[0.25, 0.5, 0.125, 1.0]], dtype=np.float32)
    out = mean_wrapper(_bf16(a), [-1], False, dtype="torch.float32")
    assert out.nbx_dtype == NBXDtype.float32
    assert np.allclose(out.numpy().ravel(), a.mean(-1))


def test_an_unknown_dtype_is_refused():
    from nbxtensor.ops.reduction_wrappers import sum_wrapper
    with pytest.raises(RuntimeError, match="not a known NBXDtype"):
        sum_wrapper(_bf16(np.ones((2, 2), np.float32)), [-1], False, dtype="torch.float8_e9m9")
