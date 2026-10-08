"""Integer floor and trunc division widen through float64 where the backend has it, through float32 where it has none
(Metal), exact below 2^24 and refused by name above.

Why: the wrappers widened integer operands to float64 unconditionally; on Metal a float64 element is misread, not
computed (`backend_has_fp64`, 2026-09-28). Seen red on Metal (wrong quotients) before the capability-driven widening."""
import numpy as np
import pytest
from neurobrix.kernels import wrappers as W
from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator, _detect_gpu_backend, backend_has_fp64


@pytest.fixture(autouse=True)
def _declared(host_backend_fp64):
    """The capability as a run declares it, from this host's profile."""


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


def test_a_large_integer_division_is_refused_by_name_where_float64_is_missing():
    _dev()
    if backend_has_fp64():
        pytest.skip("the backend has float64: no refusal on its path")
    a = np.array([1 << 25, 3, 5], dtype=np.int64)
    with pytest.raises(RuntimeError, match="2\\^24"):
        W.floor_divide_wrapper(NBXTensor.from_numpy(a), NBXTensor.from_numpy(np.array([2, 2, 2], dtype=np.int64)))
