"""Unit test — batched bmm launch (P-KERNEL-LAUNCH-HYGIENE B1).

`wrappers.bmm` used to launch per-batch Python loops of `matmul_kernel`
(prefill) and per-batch x per-row `mv_wrapper` loops (decode M<=4) — on the
deterministic decode path (`_math_attention`, batch=B*H, M=1) that was
thousands of kernel launches per generated token, pure launch-bound. It now
issues ONE batched 3D-grid launch through the existing `baddbmm_kernel`
(HAS_BIAS=False), whose K-loop body is a line-for-line mirror of
`matmul_kernel`.

Parity contract validated here:
  - exactness tests use integer-valued inputs: every product and partial
    sum is exactly representable in the fp32 accumulator, so the result is
    reduction-order independent — the batched kernel must match the torch
    reference BIT-EXACTLY regardless of the autotune config. This catches
    any indexing / masking / stride / batch-offset bug deterministically.
  - real-valued tests bound the fp-reassociation drift vs a float64 torch
    reference (the mv->tl.dot decode change is reassociation-only; the
    2026-07 captured worst case vs the looped path was max_rel 1.1e-4).
  - a launch-shape guard asserts bmm no longer calls the per-row mv loop.

Runnable two ways:
  - pytest:  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_bmm_batched.py -v
  - script:  PYTHONPATH=src python3 tests/unit/kernels/test_bmm_batched.py
"""
from __future__ import annotations

try:
    import pytest
except ModuleNotFoundError:  # script-mode under a pytest-less GPU venv
    class _NoPytest:
        class mark:
            @staticmethod
            def parametrize(*a, **k):
                return lambda fn: fn

        @staticmethod
        def skip(*a, **k):
            raise SystemExit(0)

    pytest = _NoPytest()  # type: ignore

import contextlib

import numpy as np
import torch

from neurobrix.kernels import wrappers as w
from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, nbx_to_torch


class _MockProfile:
    def __init__(self, has_native_bf16: bool):
        self.has_native_bf16 = has_native_bf16


@contextlib.contextmanager
def _hw(has_native_bf16: bool):
    """Set the wrapper hardware flag for a test, restore after."""
    saved_flag = w._NBX_HAS_NATIVE_BF16
    saved_prof = w._NBX_HW_PROFILE
    try:
        w.set_hardware_profile(_MockProfile(has_native_bf16))
        yield
    finally:
        w._NBX_HAS_NATIVE_BF16 = saved_flag
        w._NBX_HW_PROFILE = saved_prof


_NBX_DT = {"fp16": NBXDtype.float16, "bf16": NBXDtype.bfloat16,
           "fp32": NBXDtype.float32}


def _int_valued(shape, dtype: str, rng) -> NBXTensor:
    """Random integer-valued tensor in [-8, 8) — exactly representable in
    fp16/bf16/fp32; products/sums stay exact in the fp32 accumulator."""
    arr = rng.integers(-8, 8, size=shape).astype(np.float32)
    t = NBXTensor.from_numpy(np.ascontiguousarray(arr))
    return t if dtype == "fp32" else t.to(_NBX_DT[dtype])


def _real_valued(shape, dtype: str, rng) -> NBXTensor:
    arr = rng.standard_normal(shape).astype(np.float32)
    t = NBXTensor.from_numpy(np.ascontiguousarray(arr))
    return t if dtype == "fp32" else t.to(_NBX_DT[dtype])


def _torch_of(t: NBXTensor) -> torch.Tensor:
    return nbx_to_torch(t).cpu()


def _skip_no_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


# (B, M, K, N) — decode M=1/M=4 boundary, M=5 (former loop path), odd dims.
_SHAPES = [
    (32, 1, 64, 128),   # decode: attention Q@K^T class (batch=B*H)
    (32, 1, 64, 13),    # decode, odd N
    (8, 4, 32, 96),     # M<=4 boundary
    (8, 5, 32, 96),     # M>4 boundary (former per-batch matmul loop)
    (6, 37, 41, 53),    # odd everything: masks + %-wrap coverage
    (8, 128, 96, 64),   # prefill class
    (1, 64, 64, 64),    # single batch
]


@pytest.mark.parametrize("hw_bf16", [False, True])
@pytest.mark.parametrize("dtype", ["fp16", "fp32", "bf16"])
@pytest.mark.parametrize("shape", _SHAPES)
def test_bmm_exact_integer_valued(shape, dtype, hw_bf16):
    """Bit-exact vs torch on integer-valued inputs, any autotune config."""
    _skip_no_gpu()
    B, M, K, N = shape
    rng = np.random.default_rng(B * 1000 + M * 100 + K + N)
    with _hw(hw_bf16):
        a = _int_valued((B, M, K), dtype, rng)
        b = _int_valued((B, K, N), dtype, rng)
        got = _torch_of(w.bmm(a, b)).float()
    ref = torch.matmul(_torch_of(a).float(), _torch_of(b).float())
    assert got.shape == ref.shape
    assert torch.equal(got, ref), (
        f"bmm {shape} {dtype} hw_bf16={hw_bf16}: integer-valued result not "
        f"bit-exact — max|d|={(got - ref).abs().max().item()}"
    )


@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
@pytest.mark.parametrize("shape", _SHAPES)
def test_bmm_real_valued_tolerance(shape, dtype):
    """Reassociation-bounded drift vs float64 torch reference."""
    _skip_no_gpu()
    B, M, K, N = shape
    rng = np.random.default_rng(B + M + K + N)
    with _hw(False):  # pre-Ampere semantics (fp32 accumulate, fp32 store)
        a = _real_valued((B, M, K), dtype, rng)
        b = _real_valued((B, K, N), dtype, rng)
        got = _torch_of(w.bmm(a, b)).double()
    ref = torch.matmul(_torch_of(a).double(), _torch_of(b).double())
    # fp16 inputs: element error ~2^-11 per product, K-summed in fp32.
    # fp32 inputs: fp32 accumulation vs float64 reference.
    tol = 1e-2 if dtype == "fp16" else 1e-4
    assert torch.allclose(got, ref, rtol=tol, atol=tol * np.sqrt(K)), (
        f"bmm {shape} {dtype}: max|d|={(got - ref).abs().max().item()}"
    )


def test_bmm_mixed_f32xf16_promote_b():
    """fp32 activation x fp16 weight (pre-Ampere PROMOTE_B path)."""
    _skip_no_gpu()
    rng = np.random.default_rng(7)
    with _hw(False):
        a = _int_valued((4, 64, 48), "fp32", rng)
        b = _int_valued((4, 48, 80), "fp16", rng)
        got = _torch_of(w.bmm(a, b)).float()
    ref = torch.matmul(_torch_of(a).float(), _torch_of(b).float())
    assert torch.equal(got, ref)


def test_bmm_no_per_row_launch_loop():
    """Launch-hygiene guard: the decode path (M=1, B>1) must NOT fall back
    to the per-batch x per-row mv loop (B1 regression)."""
    _skip_no_gpu()
    calls = {"n": 0}
    orig = w.mv_wrapper

    def _counting_mv(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    w.mv_wrapper = _counting_mv
    try:
        rng = np.random.default_rng(3)
        with _hw(False):
            a = _real_valued((32, 1, 64), "fp16", rng)
            b = _real_valued((32, 64, 128), "fp16", rng)
            w.bmm(a, b)
    finally:
        w.mv_wrapper = orig
    assert calls["n"] == 0, (
        f"bmm(M=1) called mv_wrapper {calls['n']} times — the per-row "
        f"launch loop is back (P-KERNEL-LAUNCH-HYGIENE B1 regression)"
    )


def test_bmm_grid_z_chunking():
    """Collapsed batches beyond gridDim.z (65535) launch in chunks — the
    result must stay exact across the chunk boundary (integer-valued
    inputs, tiny inner dims)."""
    _skip_no_gpu()
    B, M, K, N = 65537, 1, 8, 8  # two chunks: 65535 + 2
    rng = np.random.default_rng(65537)
    with _hw(False):
        a = _int_valued((B, M, K), "fp16", rng)
        b = _int_valued((B, K, N), "fp16", rng)
        got = _torch_of(w.bmm(a, b)).float()
    ref = torch.matmul(_torch_of(a).float(), _torch_of(b).float())
    assert got.shape == ref.shape
    # Whole-tensor equality covers both sides of the 65535 boundary.
    assert torch.equal(got, ref), (
        f"chunked bmm diverges: max|d|={(got - ref).abs().max().item()}"
    )


@pytest.mark.parametrize("bias_kind,alpha,beta", [
    ("3d", 1.0, 1.0),
    ("2d", 2.0, 0.5),
    ("row", 0.5, 2.0),
    ("b1n", 1.0, 2.0),   # [B, 1, N] — M-broadcast (stride(-2) must zero)
    ("m1", 2.0, 1.0),    # [M, 1]    — N-broadcast (stride(1) must zero)
])
def test_baddbmm_bias_and_scaling(bias_kind, alpha, beta):
    """baddbmm keeps its bias epilogue (HAS_BIAS=True) after the B1 body
    alignment on matmul_kernel. Integer-valued inputs in [-4, 4) and
    power-of-two alpha/beta keep every value exactly representable in the
    fp16 OUTPUT (out dtype = batch1.dtype; |result| <= 2*16*48+8 < 2048)
    -> bit-exact vs torch."""
    _skip_no_gpu()
    B, M, K, N = 4, 64, 48, 80
    rng = np.random.default_rng(11)

    def _small(shape):
        arr = rng.integers(-4, 4, size=shape).astype(np.float32)
        return NBXTensor.from_numpy(
            np.ascontiguousarray(arr)).to(NBXDtype.float16)

    b1 = _small((B, M, K))
    b2 = _small((B, K, N))
    shape = {"3d": (B, M, N), "2d": (M, N), "row": (1, M, N),
             "b1n": (B, 1, N), "m1": (M, 1)}[bias_kind]
    bias = _small(shape)
    got = _torch_of(
        w.baddbmm_wrapper(bias, b1, b2, beta=beta, alpha=alpha)).float()
    ref = torch.baddbmm(_torch_of(bias).float().expand(B, M, N),
                        _torch_of(b1).float(), _torch_of(b2).float(),
                        beta=beta, alpha=alpha)
    assert got.shape == ref.shape
    assert torch.equal(got, ref), (
        f"baddbmm bias={bias_kind} a={alpha} b={beta}: "
        f"max|d|={(got - ref).abs().max().item()}"
    )


if __name__ == "__main__":
    for shape in _SHAPES:
        for dtype in ("fp16", "fp32", "bf16"):
            for hw in (False, True):
                test_bmm_exact_integer_valued(shape, dtype, hw)
        for dtype in ("fp16", "fp32"):
            test_bmm_real_valued_tolerance(shape, dtype)
    test_bmm_mixed_f32xf16_promote_b()
    test_bmm_no_per_row_launch_loop()
    test_bmm_grid_z_chunking()
    for kind, al, be in (("3d", 1.0, 1.0), ("2d", 2.0, 0.5), ("row", 0.5, 2.0),
                         ("b1n", 1.0, 2.0), ("m1", 2.0, 1.0)):
        test_baddbmm_bias_and_scaling(kind, al, be)
    print("PASS")
