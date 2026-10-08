"""Integer floor and trunc division widen through float64 where the backend has it, through float32 where it has none
(Metal), exact below 2^24 and refused by name above.

Why: the wrappers widened integer operands to float64 unconditionally; on Metal a float64 element is misread, not
computed (`backend_has_fp64`, 2026-09-28). Seen red on Metal (wrong quotients) before the capability-driven widening."""
import numpy as np
import pytest
from neurobrix.kernels import wrappers as W
from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator, _detect_gpu_backend, backend_has_fp64


def _dev():
    if _detect_gpu_backend() is None:
        pytest.skip("no device")


def test_small_integer_quotients_are_exact_on_every_backend():
    _dev()
    a = np.arange(-5000, 5000, dtype=np.int64) * 977
    b = np.full_like(a, 13)
    for name, fn, ref in (("floor", W.floor_divide_wrapper, np.floor_divide(a, b)),
                          ("trunc", lambda x, y: W.div(x, y, rounding_mode="trunc"), np.trunc(a / b).astype(np.int64))):
        out = fn(NBXTensor.from_numpy(a), NBXTensor.from_numpy(b)); DeviceAllocator.device_synchronize()
        got = out.numpy().astype(np.int64)
        assert np.array_equal(got, ref), f"{name}: {int((got != ref).sum())} wrong quotients of {a.size}"


@pytest.mark.parametrize("dtype", [np.int64, np.int32])
def test_integer_quotients_are_exact_at_every_magnitude_and_sign(dtype):
    """Two integer operands divide in the integers: exact above 2^24 too, where a float32 round-trip
    is not, with ATen's floor and trunc on every sign; a Python int divisor and a broadcast divisor."""
    _dev()
    rng = np.random.default_rng(0)
    hi = (1 << 40) if dtype == np.int64 else (1 << 30)
    a = rng.integers(-hi, hi, 20000).astype(dtype)
    b = (rng.integers(1, 5000, 20000) * rng.choice([-1, 1], 20000)).astype(dtype)
    A, B = NBXTensor.from_numpy(a), NBXTensor.from_numpy(b)
    floor = np.floor_divide(a, b)
    trunc = (np.sign(a) * np.sign(b) * (np.abs(a) // np.abs(b))).astype(dtype)
    for name, got, ref in (("floor", W.floor_divide_wrapper(A, B), floor),
                           ("trunc", W.div(A, B, rounding_mode="trunc"), trunc),
                           ("floor by -7", W.floor_divide_wrapper(A, -7), np.floor_divide(a, -7)),
                           ("floor by a row", W.floor_divide_wrapper(NBXTensor.from_numpy(a.reshape(100, 200)),
                                                                     NBXTensor.from_numpy(b[:200])),
                            np.floor_divide(a.reshape(100, 200), b[:200]))):
        DeviceAllocator.device_synchronize()
        g = got.numpy()
        assert g.dtype == dtype and np.array_equal(g, ref), f"{name}: {int((g != ref).sum())} wrong of {ref.size}"


def test_an_integer_tensor_widened_beside_a_float_is_refused_by_name_above_2_24_where_float64_is_missing():
    _dev()
    if backend_has_fp64():
        pytest.skip("the backend has float64: no refusal on its path")
    with pytest.raises(RuntimeError, match="2\\^24"):
        W._int_division_widening_dtype(NBXTensor.from_numpy(np.array([1 << 25, 3, 5], dtype=np.int64)), 2.0)
