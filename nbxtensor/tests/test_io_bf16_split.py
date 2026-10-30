"""The loader's float -> bfloat16 rounding split across its workers gives the bytes of one call (nbxtensor.io.safetensors)."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from nbxtensor.dtype.convert import f32_to_bf16_bits
from nbxtensor.io.safetensors import PAGE, spans, to_bf16_bits


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_the_split_rounding_is_the_one_call(dtype):
    n = 8 * PAGE + 37                                          # several spans, a ragged tail
    a = (np.random.default_rng(0).standard_normal(n) * 300).astype(dtype)
    a[[1, n // 2, n - 1]] = [np.nan, np.inf, -np.inf]
    with ThreadPoolExecutor(4) as pool:
        assert len(spans(a.size, a.itemsize, 1, 4)) == 4, "the work must actually be split"
        got = to_bf16_bits(a.reshape(-1, 1), (4, pool))
    assert got.shape == (a.size, 1) and got.dtype == np.uint16
    assert got.tobytes() == f32_to_bf16_bits(a.astype(np.float32)).tobytes()
