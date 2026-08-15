"""Unit test — device-scalar binary path parity (device-scalar increment).

`_prepare_binary` routes a 0-d CUDA NBXTensor scalar operand to the
`*_scalar_dev` kernels ("dev" marker) instead of a host `.item()`
extraction; the dev kernels reproduce the host binding arithmetic
in-register (load -> f64 [-> *alpha] -> f32). This removed 1,196 host
syncs per Ming t2i request and made the denoiser bucket replay-eligible
(2026-08-15).

Contract validated here:
  - ACTIVATION PROOF first: `_prepare_binary` must return the "dev"
    marker for a true 0-d CUDA scalar. A byte-equal result without this
    proof is vacuous — both historical false-greens (numpy
    ascontiguousarray flattening 0-d to (1,); NBXTensor.device
    returning self) made the equivalence tests pass while the dev path
    never ran (feedback_byte_gate_needs_activation_proof).
  - BYTE parity: add/mul/div/sub/add(alpha=2) with the scalar passed as
    a host Python value vs as a 0-d device tensor must produce
    byte-identical outputs (same dtype, same shape, same bytes) across
    x in {fp32, fp16} and scalar dtypes {fp32, int64, fp16, fp64}.
  - Cold-path guard: consumers that host-extract on "dev"
    (maximum/minimum/remainder/bitwise_and) stay byte-identical too.

Runnable two ways:
  - pytest:  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_scalar_dev_parity.py -v
  - script:  PYTHONPATH=src python3 tests/unit/kernels/test_scalar_dev_parity.py
"""
from __future__ import annotations

try:
    import pytest
except ModuleNotFoundError:  # script-mode under a pytest-less GPU venv
    pytest = None

import numpy as np


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if pytest is not None:
    pytestmark = pytest.mark.skipif(
        not _cuda_available(), reason="CUDA device required")


def _dev0d(value, np_dtype):
    """True 0-d CUDA NBXTensor. numpy ascontiguousarray promotes 0-d to
    (1,), so build a shape-() view over a 1-element upload."""
    from neurobrix.kernels.nbx_tensor import NBXTensor
    holder = NBXTensor.from_numpy(np.array([value], dtype=np_dtype))
    return NBXTensor.from_raw(holder._data_ptr, (), holder._dtype, 'cuda',
                              base=holder, device_idx=holder._device_idx)


def _download(t) -> np.ndarray:
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    buf = np.empty(t._nbytes, dtype=np.uint8)
    DeviceAllocator.memcpy(buf.ctypes.data, t.data_ptr(), t._nbytes, 2)
    return buf


def test_activation_dev_marker():
    """The routing predicate itself: 0-d CUDA scalar -> "dev" marker."""
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels import wrappers as w
    x = NBXTensor.from_numpy(np.ones(8, dtype=np.float32))
    s = _dev0d(1.0, np.float32)
    assert s.ndim == 0
    marker = w._prepare_binary(x, s)[5]
    assert marker == "dev", f"dev branch not taken (marker={marker!r})"
    # host scalar still routes to the plain scalar path
    assert w._prepare_binary(x, 1.0)[5] is True


def test_byte_parity_hot_ops():
    """add/mul/div/sub/add(alpha=2): host-scalar vs dev-scalar outputs
    byte-identical across dtype combinations."""
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels import wrappers as w
    rng = np.random.default_rng(7)
    cases = [("add", lambda a, b: w.add(a, b)),
             ("mul", w.mul),
             ("div", w.div),
             ("sub", lambda a, b: w.sub(a, b)),
             ("add_a2", lambda a, b: w.add(a, b, alpha=2.0))]
    for x_dt in (np.float32, np.float16):
        x_np = (rng.standard_normal(4096) * 3).astype(x_dt)
        x = NBXTensor.from_numpy(x_np)
        for s_dt, s_val in ((np.float32, 0.4571), (np.int64, 7),
                            (np.float16, 999.25), (np.float64, 1.0 / 3.0)):
            s_dev = _dev0d(s_val, s_dt)
            s_host = np.array(s_val, dtype=s_dt).item()
            for name, fn in cases:
                rh, rd = fn(x, s_host), fn(x, s_dev)
                label = f"{name} x={x_dt.__name__} s={s_dt.__name__}"
                assert (rh.nbx_dtype, rh.shape) == (rd.nbx_dtype, rd.shape), \
                    (f"{label}: host={rh.nbx_dtype}{rh.shape} "
                     f"dev={rd.nbx_dtype}{rd.shape}")
                assert np.array_equal(_download(rh), _download(rd)), \
                    f"{label}: bytes differ"


def test_byte_parity_cold_ops():
    """maximum/minimum/remainder host-extract on "dev" — exact old
    behavior preserved."""
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels import wrappers as w
    x = NBXTensor.from_numpy(np.arange(-8, 8, dtype=np.float32))
    s = _dev0d(2.5, np.float32)
    for name, fn in (("maximum", w.maximum_wrapper),
                     ("minimum", w.minimum_wrapper),
                     ("remainder", w.remainder_wrapper)):
        rh, rd = fn(x, 2.5), fn(x, s)
        assert rh.nbx_dtype == rd.nbx_dtype, name
        assert np.array_equal(_download(rh), _download(rd)), name
    xb = NBXTensor.from_numpy(np.array([0, 1, 1, 0], dtype=np.bool_))
    sb = _dev0d(True, np.bool_)
    rh = w.bitwise_and_wrapper(xb, True)
    rd = w.bitwise_and_wrapper(xb, sb)
    assert np.array_equal(_download(rh), _download(rd)), "bitwise_and"


if __name__ == "__main__":
    if not _cuda_available():
        raise SystemExit("CUDA device required")
    test_activation_dev_marker()
    test_byte_parity_hot_ops()
    test_byte_parity_cold_ops()
    print("ALL PASS (activation + 40-case byte parity + cold ops)")
