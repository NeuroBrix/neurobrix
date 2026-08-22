"""CORRECTNESS ORACLE — convolution and the index/scatter family.

Completes the numeric-kernel audit opened by the attention defect. Those
two families are the last uncovered ones named in the family oracle's own
"not yet covered" note.

Two different kinds of check, deliberately:

- **conv2d** is a numeric reduction, so it gets the same treatment as the
  matmul family: an independently written float64 reference, swept
  inputs, a derived bound.
- **index / scatter / gather** are DATA MOVEMENT, not arithmetic. Their
  correct output is bit-exact by definition — there is no rounding to
  tolerate — so a bound would be the wrong instrument. They are checked
  for **exact equality** against a numpy reference, which is a stricter
  contract than any tolerance.

That distinction matters: applying a float tolerance to an indexing
kernel would let a genuinely wrong index pass whenever the values it
picked up happened to be close.

=== CONV2D BOUND, DERIVED ===

fp16 inputs (2^-11 = 4.9e-04 per value), fp32 accumulation over
C_in * kh * kw terms. At C_in=64, k=3 that is 576 terms;
4.9e-04 * sqrt(576) = 1.2e-02. **Bound 5e-02**, the same class and the
same derivation as the matmul family, with `test_bound_is_not_vacuous`
holding both ends.

Runnable two ways:
  - pytest:  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_conv_index_correctness.py -v
  - script:  PYTHONPATH=src python3 tests/unit/kernels/test_conv_index_correctness.py
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

CONV_FP16_REL_BOUND = 5e-02

_NP = {NBXDtype.float32: (np.float32, 4), NBXDtype.float16: (np.float16, 2),
       NBXDtype.int64: (np.int64, 8), NBXDtype.int32: (np.int32, 4)}


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
    return np.frombuffer(bytes(buf), dtype=dt).copy()


def _t(x):
    return NBXTensor.from_numpy(np.ascontiguousarray(x))


def _conv2d_float64(x, w, stride=1, padding=0):
    """Independent float64 convolution, written from the definition."""
    X, Wt = x.astype(np.float64), w.astype(np.float64)
    N, C, H, Wd = X.shape
    O, _, kh, kw = Wt.shape
    if padding:
        X = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    Ho = (X.shape[2] - kh) // stride + 1
    Wo = (X.shape[3] - kw) // stride + 1
    out = np.zeros((N, O, Ho, Wo), dtype=np.float64)
    for oh in range(Ho):
        for ow in range(Wo):
            patch = X[:, :, oh * stride:oh * stride + kh,
                      ow * stride:ow * stride + kw]
            out[:, :, oh, ow] = np.tensordot(patch, Wt, axes=([1, 2, 3],
                                                              [1, 2, 3]))
    return out


@pytest.mark.parametrize("N,C,H,Wd,O,k,stride,pad", [
    (1, 8, 16, 16, 16, 3, 1, 1),
    (1, 32, 32, 32, 32, 3, 1, 1),
    (2, 16, 24, 24, 8, 1, 1, 0),
    (1, 64, 16, 16, 32, 3, 2, 1),
    (1, 4, 15, 17, 6, 3, 1, 0),      # odd spatial dims, no padding
])
def test_conv2d_matches_float64(N, C, H, Wd, O, k, stride, pad) -> None:
    if not _has_gpu():
        pytest.skip("no GPU")
    rng = np.random.default_rng(C * 100 + O)
    x = (rng.standard_normal((N, C, H, Wd)) * 0.5).astype(np.float16)
    w = (rng.standard_normal((O, C, k, k)) * 0.1).astype(np.float16)
    got = _d2h(W.conv2d_wrapper(_t(x), _t(w), None,
                                stride=[stride, stride],
                                padding=[pad, pad])).astype(np.float64)
    ref = _conv2d_float64(x, w, stride, pad)
    err = float(np.abs(got - ref.reshape(-1)).max()
                / max(np.abs(ref).max(), 1e-9))
    assert err <= CONV_FP16_REL_BOUND, (
        f"conv2d N={N} C={C} {H}x{Wd} O={O} k={k} s={stride} p={pad}: "
        f"relative error {err:.3e} exceeds {CONV_FP16_REL_BOUND:.1e}")


# ======================================================================
# index / scatter / gather — EXACT, not toleranced
# ======================================================================

@pytest.mark.parametrize("rows,cols,picks", [(64, 32, 16), (1024, 8, 500),
                                             (17, 5, 17)])
def test_index_select_is_exact(rows, cols, picks) -> None:
    """Data movement has no rounding: a tolerance here would let a wrong
    index pass whenever it happened to land on a close value."""
    if not _has_gpu():
        pytest.skip("no GPU")
    rng = np.random.default_rng(rows)
    x = rng.standard_normal((rows, cols)).astype(np.float32)
    idx = rng.integers(0, rows, size=picks).astype(np.int64)
    got = _d2h(W.index_select_wrapper(_t(x), 0, _t(idx)))
    ref = x[idx].reshape(-1)
    assert np.array_equal(got, ref), (
        f"index_select rows={rows} picks={picks}: not exact "
        f"({int(np.count_nonzero(got != ref))} elements differ)")


@pytest.mark.parametrize("rows,cols", [(32, 16), (128, 4), (7, 13)])
def test_gather_is_exact(rows, cols) -> None:
    if not _has_gpu():
        pytest.skip("no GPU")
    rng = np.random.default_rng(rows + cols)
    x = rng.standard_normal((rows, cols)).astype(np.float32)
    idx = rng.integers(0, cols, size=(rows, cols)).astype(np.int64)
    got = _d2h(W.gather_wrapper(_t(x), 1, _t(idx)))
    ref = np.take_along_axis(x, idx, axis=1).reshape(-1)
    assert np.array_equal(got, ref), (
        f"gather {rows}x{cols}: not exact "
        f"({int(np.count_nonzero(got != ref))} elements differ)")


@pytest.mark.parametrize("rows,cols", [(32, 16), (64, 8)])
def test_scatter_is_exact(rows, cols) -> None:
    if not _has_gpu():
        pytest.skip("no GPU")
    rng = np.random.default_rng(rows * 3 + cols)
    base = rng.standard_normal((rows, cols)).astype(np.float32)
    src = rng.standard_normal((rows, cols)).astype(np.float32)
    # a permutation per row keeps the scatter one-to-one, so the
    # reference is unambiguous
    idx = np.stack([rng.permutation(cols) for _ in range(rows)]).astype(np.int64)
    got = _d2h(W.scatter_wrapper(_t(base), 1, _t(idx), _t(src)))
    ref = base.copy()
    np.put_along_axis(ref, idx, src, axis=1)
    assert np.array_equal(got, ref.reshape(-1)), (
        f"scatter {rows}x{cols}: not exact "
        f"({int(np.count_nonzero(got != ref.reshape(-1)))} elements differ)")


@pytest.mark.parametrize("rows,cols,adds", [(64, 8, 32), (128, 4, 128), (256, 4, 200)])
def test_index_add_is_exact_on_disjoint_indices(rows, cols, adds) -> None:
    """Disjoint indices on purpose: with repeats the result depends on
    accumulation order and exactness is the wrong contract. Disjoint
    keeps it a pure move plus one add, which must be exact."""
    if not _has_gpu():
        pytest.skip("no GPU")
    assert adds <= rows, (
        "disjoint indices require adds <= rows; a permutation of `rows` "
        "cannot yield more than `rows` distinct values")
    rng = np.random.default_rng(rows + adds)
    base = rng.standard_normal((rows, cols)).astype(np.float32)
    idx = rng.permutation(rows)[:adds].astype(np.int64)
    src = rng.standard_normal((adds, cols)).astype(np.float32)
    got = _d2h(W.index_add_wrapper(_t(base), 0, _t(idx), _t(src)))
    ref = base.copy()
    ref[idx] += src
    assert np.array_equal(got, ref.reshape(-1)), (
        f"index_add rows={rows} adds={adds}: not exact "
        f"({int(np.count_nonzero(got != ref.reshape(-1)))} elements differ)")


def test_bound_is_not_vacuous() -> None:
    assert CONV_FP16_REL_BOUND < 3.4e-01 / 5, "conv bound too loose"
    assert CONV_FP16_REL_BOUND >= 4.9e-04, "conv bound below one fp16 unit"


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
            print(f"  ERROR {label}: {type(e).__name__}: {str(e)[:60]}")

    print("=== conv2d vs float64 ===")
    for a in [(1, 8, 16, 16, 16, 3, 1, 1), (1, 32, 32, 32, 32, 3, 1, 1),
              (2, 16, 24, 24, 8, 1, 1, 0), (1, 64, 16, 16, 32, 3, 2, 1),
              (1, 4, 15, 17, 6, 3, 1, 0)]:
        run(f"conv2d {a}", lambda a=a: test_conv2d_matches_float64(*a))
    print("=== index / scatter / gather — EXACT ===")
    for r, c, p in [(64, 32, 16), (1024, 8, 500), (17, 5, 17)]:
        run(f"index_select {r}x{c} picks={p}",
            lambda r=r, c=c, p=p: test_index_select_is_exact(r, c, p))
    for r, c in [(32, 16), (128, 4), (7, 13)]:
        run(f"gather {r}x{c}", lambda r=r, c=c: test_gather_is_exact(r, c))
    for r, c in [(32, 16), (64, 8)]:
        run(f"scatter {r}x{c}", lambda r=r, c=c: test_scatter_is_exact(r, c))
    for r, c, a in [(64, 8, 32), (128, 4, 128), (256, 4, 200)]:
        run(f"index_add {r}x{c} adds={a}",
            lambda r=r, c=c, a=a: test_index_add_is_exact_on_disjoint_indices(r, c, a))
    run("bound not vacuous", test_bound_is_not_vacuous)
    print(f"\n{'ALL GREEN' if not fails else f'{fails} CORRECTNESS FAILURE(S)'}")
    raise SystemExit(1 if fails else 0)
