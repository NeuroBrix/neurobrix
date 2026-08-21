"""CORRECTNESS ORACLE — the numeric kernel family beyond attention.

Companion to `test_numeric_correctness_oracle.py`, which covers
attention. That file exists because a systematically wrong kernel passes
every byte gate: the Triton attention kernel served head dimension 128 at
a relative error of 3.4e-01 for weeks, through every full-zoo gate, and
surfaced only by accident.

**The expectation going in is that attention is not the only one.** This
file audits the rest of the numeric family against independently written
float64 references, kernel by kernel, and says of each whether it is
right — not whether it repeats itself.

=== HOW A BOUND IS DERIVED HERE ===

Not chosen to make a test pass. For each kernel:

  unit    fp16 carries 2^-11 = 4.9e-04 of relative error per rounded
          value; fp32 carries 2^-24 = 6.0e-08.
  depth   how many rounded values are summed. A matmul over K terms in
          an fp32 accumulator has the inputs' rounding dominate, growing
          with sqrt(K) in the worst realistic case rather than K.
  bound   unit x sqrt(depth), then rounded up to the next convenient
          value, then checked against the *measured* error on the same
          shape so the margin is stated rather than assumed.

`test_bounds_are_not_vacuous` asserts every bound is comfortably tighter
than the attention defect it must be able to catch (3.4e-01), so no bound
can be quietly loosened into meaninglessness.

Runnable two ways:
  - pytest:  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_numeric_correctness_family.py -v
  - script:  PYTHONPATH=src python3 tests/unit/kernels/test_numeric_correctness_family.py
"""
from __future__ import annotations

import ctypes

import numpy as np

try:
    import pytest
except ModuleNotFoundError:  # script-mode under a pytest-less GPU venv
    class _NoPytest:  # pragma: no cover - shim
        class mark:
            @staticmethod
            def parametrize(*a, **k):
                return lambda fn: fn

        @staticmethod
        def skip(*a, **k):
            raise SystemExit(0)

    pytest = _NoPytest()  # type: ignore[assignment]

from neurobrix.kernels import wrappers as W
from neurobrix.kernels.nbx_tensor import DeviceAllocator, NBXDtype, NBXTensor

_NP = {NBXDtype.float32: (np.float32, 4), NBXDtype.float16: (np.float16, 2)}

# --- declared bounds, derived in the module docstring -------------------
#
# matmul family: fp16 inputs, fp32 accumulator, K up to 4096.
#   4.9e-04 * sqrt(4096) = 3.1e-02 worst case; measured well below.
BOUND_MATMUL_FP16 = 5e-02
# elementwise: one rounded op, no accumulation.
BOUND_ELEMENTWISE_FP16 = 2e-03
# reductions over N: 4.9e-04 * sqrt(N), N up to 4096 -> 3.1e-02.
BOUND_REDUCTION_FP16 = 5e-02
# normalisations: a reduction followed by a division; same class.
BOUND_NORM_FP16 = 5e-02


def _has_gpu() -> bool:
    try:
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


def _d2h(t):
    dt, sz = _NP[t._dtype]
    n = t.numel()
    buf = (ctypes.c_char * (n * sz))()
    DeviceAllocator.memcpy(ctypes.addressof(buf), t.data_ptr(), n * sz, kind=2)
    return np.frombuffer(bytes(buf), dtype=dt).astype(np.float64)


def _rel(got, ref):
    return float(np.abs(got - ref.reshape(-1)).max() / max(np.abs(ref).max(), 1e-9))


def _t(x):
    return NBXTensor.from_numpy(np.ascontiguousarray(x))


# ======================================================================
# matmul family
# ======================================================================

@pytest.mark.parametrize("M,K,N", [(1, 2048, 2048), (32, 512, 512),
                                   (128, 128, 128), (1, 4096, 1024),
                                   (23, 257, 129)])
def test_mm_is_correct(M, K, N) -> None:
    if not _has_gpu():
        pytest.skip("no GPU")
    rng = np.random.default_rng(M * 7 + K)
    a = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((K, N)) * 0.1).astype(np.float16)
    got = _d2h(W.mm(_t(a), _t(b)))
    ref = a.astype(np.float64) @ b.astype(np.float64)
    err = _rel(got, ref)
    assert err <= BOUND_MATMUL_FP16, (
        f"mm M={M} K={K} N={N}: {err:.3e} > {BOUND_MATMUL_FP16:.1e}")


@pytest.mark.parametrize("B,M,K,N", [(4, 1, 128, 128), (32, 1, 256, 128),
                                     (2, 23, 64, 64), (8, 16, 512, 64)])
def test_bmm_is_correct(B, M, K, N) -> None:
    if not _has_gpu():
        pytest.skip("no GPU")
    rng = np.random.default_rng(B * 13 + K)
    a = (rng.standard_normal((B, M, K)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((B, K, N)) * 0.1).astype(np.float16)
    got = _d2h(W.bmm(_t(a), _t(b)))
    ref = a.astype(np.float64) @ b.astype(np.float64)
    err = _rel(got, ref)
    assert err <= BOUND_MATMUL_FP16, (
        f"bmm B={B} M={M} K={K} N={N}: {err:.3e} > {BOUND_MATMUL_FP16:.1e}")


# ======================================================================
# elementwise
# ======================================================================

@pytest.mark.parametrize("fn,ref_fn,name", [
    (lambda x: W.silu(x), lambda x: x / (1.0 + np.exp(-x)), "silu"),
    (lambda x: W.gelu(x),
     lambda x: 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3))),
     "gelu-tanh"),
])
@pytest.mark.parametrize("n", [1024, 4096])
def test_elementwise_is_correct(fn, ref_fn, name, n) -> None:
    if not _has_gpu():
        pytest.skip("no GPU")
    rng = np.random.default_rng(n)
    x = (rng.standard_normal((1, n)) * 2.0).astype(np.float16)
    got = _d2h(fn(_t(x)))
    ref = ref_fn(x.astype(np.float64))
    err = _rel(got, ref)
    # gelu has two conventions (exact erf vs tanh approximation); a
    # mismatch here is a CONVENTION difference, reported as such rather
    # than silently tolerated.
    assert err <= BOUND_ELEMENTWISE_FP16, (
        f"{name} n={n}: {err:.3e} > {BOUND_ELEMENTWISE_FP16:.1e} "
        f"(if this is gelu, check erf vs tanh convention before the kernel)")


# ======================================================================
# reductions and normalisations
# ======================================================================

@pytest.mark.parametrize("n", [128, 1024, 4096, 151936])
def test_softmax_is_correct(n) -> None:
    if not _has_gpu():
        pytest.skip("no GPU")
    rng = np.random.default_rng(n)
    x = (rng.standard_normal((1, n)) * 3.0).astype(np.float16)
    got = _d2h(W.softmax(_t(x), dim=-1))
    xd = x.astype(np.float64)
    e = np.exp(xd - xd.max(-1, keepdims=True))
    ref = e / e.sum(-1, keepdims=True)
    err = _rel(got, ref)
    assert err <= BOUND_REDUCTION_FP16, (
        f"softmax n={n}: {err:.3e} > {BOUND_REDUCTION_FP16:.1e}")


@pytest.mark.parametrize("n", [128, 2048, 4096])
def test_rms_norm_is_correct(n) -> None:
    if not _has_gpu():
        pytest.skip("no GPU")
    rng = np.random.default_rng(n + 1)
    x = (rng.standard_normal((1, n)) * 1.5).astype(np.float16)
    w = (rng.standard_normal((n,)) * 0.5 + 1.0).astype(np.float16)
    eps = 1e-6
    got = _d2h(W.rms_norm(_t(x), _t(w), eps=eps))
    xd, wd = x.astype(np.float64), w.astype(np.float64)
    ref = xd / np.sqrt((xd ** 2).mean(-1, keepdims=True) + eps) * wd
    err = _rel(got, ref)
    assert err <= BOUND_NORM_FP16, (
        f"rms_norm n={n}: {err:.3e} > {BOUND_NORM_FP16:.1e}")


@pytest.mark.parametrize("n", [128, 2048])
def test_cumsum_is_correct(n) -> None:
    if not _has_gpu():
        pytest.skip("no GPU")
    rng = np.random.default_rng(n + 2)
    x = (rng.random((1, n)) * 0.01).astype(np.float32)
    got = _d2h(W.cumsum_wrapper(_t(x), dim=-1))
    ref = np.cumsum(x.astype(np.float64), axis=-1)
    err = _rel(got, ref)
    assert err <= BOUND_REDUCTION_FP16, (
        f"cumsum n={n}: {err:.3e} > {BOUND_REDUCTION_FP16:.1e}")


@pytest.mark.parametrize("shape", [(1, 1024), (4, 2048), (32, 128)])
def test_sum_is_correct(shape) -> None:
    if not _has_gpu():
        pytest.skip("no GPU")
    rng = np.random.default_rng(shape[1])
    x = (rng.standard_normal(shape) * 0.5).astype(np.float16)
    got = _d2h(W.sum_wrapper(_t(x), dim=-1))
    ref = x.astype(np.float64).sum(-1)
    err = _rel(got, ref)
    assert err <= BOUND_REDUCTION_FP16, (
        f"sum {shape}: {err:.3e} > {BOUND_REDUCTION_FP16:.1e}")


def test_bounds_are_not_vacuous() -> None:
    """Every bound must be able to catch a defect of the size that
    started this whole audit (attention at 3.4e-01)."""
    for name, b in (("matmul", BOUND_MATMUL_FP16),
                    ("elementwise", BOUND_ELEMENTWISE_FP16),
                    ("reduction", BOUND_REDUCTION_FP16),
                    ("norm", BOUND_NORM_FP16)):
        assert b < 3.4e-01 / 5, f"{name} bound {b} too loose to catch the attention-class defect"
        assert b >= 4.9e-04, f"{name} bound {b} is below one fp16 rounding unit"


if __name__ == "__main__":  # script mode
    if not _has_gpu():
        raise SystemExit("no GPU")
    fails = 0

    def run(label, fn):
        global fails
        try:
            fn()
            print(f"  ok    {label}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  {label}\n        {str(e).splitlines()[0]}")
        except Exception as e:
            fails += 1
            print(f"  ERROR {label}: {type(e).__name__}: {str(e)[:70]}")

    print("=== matmul family ===")
    for M, K, N in [(1, 2048, 2048), (32, 512, 512), (128, 128, 128),
                    (1, 4096, 1024), (23, 257, 129)]:
        run(f"mm {M}x{K}x{N}", lambda M=M, K=K, N=N: test_mm_is_correct(M, K, N))
    for B, M, K, N in [(4, 1, 128, 128), (32, 1, 256, 128), (2, 23, 64, 64),
                       (8, 16, 512, 64)]:
        run(f"bmm {B}x{M}x{K}x{N}",
            lambda B=B, M=M, K=K, N=N: test_bmm_is_correct(B, M, K, N))
    print("=== elementwise ===")
    for name, fn, rf in (("silu", lambda x: W.silu(x), lambda x: x / (1 + np.exp(-x))),
                         ("gelu-tanh", lambda x: W.gelu(x),
                          lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))))):
        for n in (1024, 4096):
            run(f"{name} n={n}",
                lambda fn=fn, rf=rf, name=name, n=n:
                test_elementwise_is_correct(fn, rf, name, n))
    print("=== reductions / norms ===")
    for n in (128, 1024, 4096, 151936):
        run(f"softmax n={n}", lambda n=n: test_softmax_is_correct(n))
    for n in (128, 2048, 4096):
        run(f"rms_norm n={n}", lambda n=n: test_rms_norm_is_correct(n))
    for n in (128, 2048):
        run(f"cumsum n={n}", lambda n=n: test_cumsum_is_correct(n))
    for s in ((1, 1024), (4, 2048), (32, 128)):
        run(f"sum {s}", lambda s=s: test_sum_is_correct(s))
    run("bounds not vacuous", test_bounds_are_not_vacuous)
    print(f"\n{'ALL GREEN' if not fails else f'{fails} CORRECTNESS FAILURE(S)'}")
    raise SystemExit(1 if fails else 0)
