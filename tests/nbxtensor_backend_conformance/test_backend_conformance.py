"""The backend conformance suite (docs/nbxtensor/BACKENDS.md): what any nbxtensor backend must do, checked against a
numpy oracle — allocation and copies, one kernel of each family, views and strides, a README example. Parametrized by
backend: the backend in force on this machine runs; the others are skipped with the reason. It imports the library
only: a backend is proven without the application around it.

    python -m pytest tests/nbxtensor_backend_conformance
"""
from __future__ import annotations

import numpy as np
import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.tensor import NBXTensor

BACKENDS = ("cuda", "metal", "cpu")
_NOT_HERE = {"cpu": "nbxtensor 0.0.1 has no CPU kernel backend (README, Hardware): host work runs with numpy"}


def _in_force() -> str:
    from nbxtensor.backend.runtime import _detect_gpu_backend
    return _detect_gpu_backend()


@pytest.fixture(params=BACKENDS)
def backend(request):
    name = request.param
    if name in _NOT_HERE:
        pytest.skip(_NOT_HERE[name])
    here = _in_force()
    if here != name:
        pytest.skip(f"the backend in force on this machine is {here!r}")
    return name


RNG = np.random.default_rng(20261029)


def _t(a):
    return NBXTensor.from_numpy(np.ascontiguousarray(a))


def _close(got, want, rtol, atol):
    np.testing.assert_allclose(np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64),
                               rtol=rtol, atol=atol)


# --- allocation and copies ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int32, np.int64, np.uint8, np.bool_])
def test_a_host_array_goes_to_the_device_and_back_bit_for_bit(backend, dtype):
    a = (RNG.standard_normal((37, 19)) * 100).astype(dtype)
    t = _t(a)
    assert tuple(t.shape) == a.shape
    assert t.numpy().tobytes() == a.tobytes()


def test_a_device_copy_is_its_own_memory(backend):
    a = RNG.standard_normal((64,)).astype(np.float32)
    t = _t(a)
    c = t.clone()
    assert c.data_ptr() != t.data_ptr() and c.numpy().tobytes() == a.tobytes()


def test_zeros_are_zeros(backend):
    z = NBXTensor.zeros((5, 7), NBXDtype.float32)          # the device in force
    assert not z.numpy().any()


# --- one kernel of each family -----------------------------------------------------------------------------------
def test_elementwise(backend):
    from nbxtensor.ops import elementwise_wrappers as E
    a, b = RNG.standard_normal((257,)).astype(np.float32), RNG.standard_normal((257,)).astype(np.float32)
    _close(E.add(_t(a), _t(b)).numpy(), a + b, 0, 0)
    _close(E.mul(_t(a), _t(b)).numpy(), a * b, 0, 0)
    _close(E.exp(_t(a)).numpy(), np.exp(a.astype(np.float64)), 1e-6, 1e-6)


def test_reduction(backend):
    from nbxtensor.ops import reduction_wrappers as R
    a = RNG.standard_normal((33, 65)).astype(np.float32)
    _close(R.sum_wrapper(_t(a), dim=1).numpy(), a.astype(np.float64).sum(1), 1e-5, 1e-5)
    _close(R.amax_wrapper(_t(a), dim=0).numpy(), a.max(0), 0, 0)


@pytest.mark.parametrize("dtype,rtol", [(np.float32, 1e-4), (np.float16, 2e-2)])
def test_matrix_products(backend, dtype, rtol):
    from nbxtensor.ops import linalg_wrappers as L
    a, b = RNG.standard_normal((48, 64)).astype(dtype), RNG.standard_normal((64, 40)).astype(dtype)
    _close(L.mm(_t(a), _t(b)).numpy(), a.astype(np.float64) @ b.astype(np.float64), rtol, rtol)
    x, y = RNG.standard_normal((3, 16, 32)).astype(dtype), RNG.standard_normal((3, 32, 8)).astype(dtype)
    _close(L.bmm(_t(x), _t(y)).numpy(), x.astype(np.float64) @ y.astype(np.float64), rtol, rtol)


def test_softmax_and_norms(backend):
    from nbxtensor.ops import attention_wrappers as A, norm_wrappers as N
    x = RNG.standard_normal((6, 128)).astype(np.float32)
    e = np.exp(x.astype(np.float64) - x.max(1, keepdims=True))
    _close(A.softmax(_t(x), -1).numpy(), e / e.sum(1, keepdims=True), 1e-5, 1e-6)
    w = RNG.standard_normal((128,)).astype(np.float32)
    rms = x.astype(np.float64) / np.sqrt((x.astype(np.float64) ** 2).mean(1, keepdims=True) + 1e-6) * w
    _close(N.rms_norm(_t(x), _t(w), eps=1e-6).numpy(), rms, 1e-5, 1e-5)
    mu, var = x.astype(np.float64).mean(1, keepdims=True), x.astype(np.float64).var(1, keepdims=True)
    _close(N.layer_norm_wrapper(_t(x), [128], _t(w), None, 1e-5).numpy(), (x - mu) / np.sqrt(var + 1e-5) * w,
           1e-5, 1e-5)


def test_convolution(backend):
    from nbxtensor.ops import conv_wrappers as C
    x = RNG.standard_normal((1, 3, 12, 12)).astype(np.float32)
    w = RNG.standard_normal((4, 3, 3, 3)).astype(np.float32)
    xp = np.pad(x.astype(np.float64), ((0, 0), (0, 0), (1, 1), (1, 1)))
    want = np.zeros((1, 4, 12, 12))
    for i in range(12):
        for j in range(12):
            want[0, :, i, j] = np.tensordot(w.astype(np.float64), xp[0, :, i:i + 3, j:j + 3], axes=([1, 2, 3], [0, 1, 2]))
    _close(C.conv2d_wrapper(_t(x), _t(w), None, 1, 1).numpy(), want, 1e-4, 1e-4)


def test_indexing(backend):
    from nbxtensor.ops import index_wrappers as I
    table = RNG.standard_normal((10, 6)).astype(np.float32)
    ids = np.array([3, 0, 9, 3], dtype=np.int64)
    _close(I.embedding(_t(table), _t(ids)).numpy(), table[ids], 0, 0)
    _close(I.index_select_wrapper(_t(table), 0, _t(ids)).numpy(), table[ids], 0, 0)
    cond = RNG.standard_normal((10, 6)) > 0
    other = np.zeros_like(table)
    _close(I.where_wrapper(_t(cond), _t(table), _t(other)).numpy(), np.where(cond, table, other), 0, 0)


# --- views and strides -------------------------------------------------------------------------------------------
def test_views_read_through_their_strides(backend):
    a = RNG.standard_normal((4, 5, 6)).astype(np.float32)
    t = _t(a)
    assert t.permute(2, 0, 1).contiguous().numpy().tobytes() == np.ascontiguousarray(a.transpose(2, 0, 1)).tobytes()
    assert t.transpose(0, 1).contiguous().numpy().tobytes() == np.ascontiguousarray(a.swapaxes(0, 1)).tobytes()
    assert t.view(20, 6).numpy().tobytes() == a.reshape(20, 6).tobytes()
    assert t[1].contiguous().numpy().tobytes() == a[1].tobytes()


# --- a README example --------------------------------------------------------------------------------------------
def test_a_readme_example_runs_on_this_backend(backend):
    import pathlib, re
    readme = pathlib.Path(__file__).resolve().parents[2] / "nbxtensor" / "README.md"
    blocks = re.findall(r"```python\n(.*?)```", readme.read_text(), re.S)
    assert blocks
    for i, block in enumerate(blocks):
        exec(compile(block, f"README.md example {i + 1}", "exec"), {})
