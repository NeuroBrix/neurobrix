"""Unit test — an fp32 GEMM on the m8n8k4 matrix unit through the hi + lo split loses nothing against the fp32 path.

Where the hardware profile declares `matrix_unit.fp32_split` (volta.yml), the DtypeEngine
(`triton/dtype.matrix_unit_representation`) carries an fp32 operand to the fp16 unit as hi + lo under a power-of-two
scale per row (A) or column (B) of each K tile: three MMAs, fp32 accumulation (ops/matmul_m8n8k4.py). Each case is
judged against an fp64 oracle computed in-test AND against the FMA path the operand took before (the unit withheld,
`tl.dot(input_precision="ieee")`): the split's error must not exceed the FMA path's, beyond four fp32 epsilons. The
cases are shapes of the census (fp32 x fp16 and fp32 x fp32 addmm, fp32 baddbmm) and the range edges (beyond fp16's
65504, far below its smallest normal, rows of 1e6 next to rows of 1e-6, elements over 24 decades, zero rows), the
error judged per row against that row's own magnitude. Removing the lo term fails every case (seen failing:
docs/reference/tensor-cores-through-linear-layouts.md). Skipped where the profile declares no split.

  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_gemm_fp32_split.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels import wrappers as W
from neurobrix.kernels.nbx_tensor import DeviceAllocator, NBXTensor


def _split_declared() -> bool:
    try:
        DeviceAllocator.set_device(0)
        return bool(((W._matrix_unit() or {}).get("fp32_split") or {}).get("mm"))
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _split_declared(), reason="the hardware profile declares no matrix_unit.fp32_split")

_SLACK = 4 * float(np.finfo(np.float32).eps)


@pytest.fixture(autouse=True, scope="module")
def _profile():
    from neurobrix.kernels import autotune_certify as Z
    Z._bind_hardware_profile()


def _nb(x):
    return NBXTensor.from_numpy(np.ascontiguousarray(x)).to("cuda:0")


def _per_row(got, ref):
    """max over rows of |got - ref| / max|ref| of that row: a small row judged against itself."""
    got, ref = got.reshape(-1, ref.shape[-1]), ref.reshape(-1, ref.shape[-1])
    return float((np.abs(got - ref).max(axis=1) / np.maximum(np.abs(ref).max(axis=1), 1e-300)).max())


def _both(run, monkeypatch):
    """(split result, its tile's split flags), FMA-path result — the second with the unit withheld."""
    calls = []
    real = W._gemm_m8n8k4
    monkeypatch.setattr(W, "_gemm_m8n8k4", lambda *a, **kw: (calls.append(a[8]), real(*a, **kw))[1])
    split = run().numpy().astype(np.float64)
    assert calls, "the m8n8k4 route was not taken"
    flags = (calls[0]["split_a"], calls[0]["split_b"])
    from neurobrix.kernels.ops import _configs as C
    monkeypatch.setattr(C, "matrix_unit", lambda: {})
    monkeypatch.setattr(W, "_matrix_unit", lambda: {})
    calls.clear()
    fma = run().numpy().astype(np.float64)
    assert not calls
    return split, flags, fma


def _judge(split, fma, ref):
    es, ef = _per_row(split, ref), _per_row(fma, ref)
    assert np.isfinite(split).all()
    assert es <= ef + _SLACK, f"split {es:.2e} vs fma {ef:.2e} (per-row relative to fp64)"


@pytest.mark.parametrize("M,K,N,w_dtype", [
    (1024, 2304, 1152, np.float16),     # Allegro: fp32 activation x fp16 weight
    (2048, 512, 512, np.float32),       # Allegro: fp32 x fp32
    (512, 5120, 1024, np.float32),      # Wan, K of the attention projections
    (256, 13824, 512, np.float32),      # Wan, the ffn down projection's K
])
def test_addmm_census_shapes(M, K, N, w_dtype, monkeypatch):
    rng = np.random.default_rng(M + K)
    a = rng.standard_normal((M, K)).astype(np.float32)
    w = (rng.standard_normal((N, K)) / np.sqrt(K)).astype(w_dtype)
    bias = rng.standard_normal(N).astype(np.float32)
    A, Wt, B = _nb(a), _nb(w).t(), _nb(bias)
    split, flags, fma = _both(lambda: W.addmm(B, A, Wt), monkeypatch)
    assert flags == (True, w_dtype == np.float32)
    _judge(split, fma, bias.astype(np.float64) + a.astype(np.float64) @ w.astype(np.float64).T)


@pytest.mark.parametrize("Z,M,K,N", [(4, 1152, 512, 1152), (8, 512, 64, 512)])
def test_baddbmm_census_shapes(Z, M, K, N, monkeypatch):
    rng = np.random.default_rng(Z * M)
    a = rng.standard_normal((Z, M, K)).astype(np.float32)
    b = (rng.standard_normal((Z, K, N)) / 8).astype(np.float32)
    bias = rng.standard_normal((Z, M, N)).astype(np.float32)
    X, Y, Bz = _nb(a), _nb(b), _nb(bias)
    split, flags, fma = _both(lambda: W.baddbmm_wrapper(Bz, X, Y, beta=0.5, alpha=2.0), monkeypatch)
    assert flags == (True, True)
    _judge(split, fma, 0.5 * bias.astype(np.float64)
           + 2.0 * np.einsum("zmk,zkn->zmn", a.astype(np.float64), b.astype(np.float64)))


_EDGES = {
    "beyond fp16 (1e6)": lambda r, s: r.standard_normal(s) * 1e6,
    "below fp16 normals (1e-8)": lambda r, s: r.standard_normal(s) * 1e-8,
    "far below (1e-30)": lambda r, s: r.standard_normal(s) * 1e-30,
    "rows 1e6 next to rows 1e-6": lambda r, s: r.standard_normal(s) * np.where(np.arange(s[0])[:, None] % 2, 1e6, 1e-6),
    "elements over 24 decades": lambda r, s: r.standard_normal(s) * 10.0 ** r.uniform(-12, 12, s),
    "zero rows": lambda r, s: r.standard_normal(s) * (np.arange(s[0])[:, None] % 3 != 0),
}


@pytest.mark.parametrize("edge", list(_EDGES))
def test_range_edges(edge, monkeypatch):
    rng = np.random.default_rng(len(edge))
    M, K, N = 512, 1024, 256
    a = _EDGES[edge](rng, (M, K)).astype(np.float32)
    w = rng.standard_normal((K, N)).astype(np.float32)
    A, Wb = _nb(a), _nb(w)
    split, flags, fma = _both(lambda: W.mm(A, Wb), monkeypatch)
    assert flags == (True, True)
    ref = a.astype(np.float64) @ w.astype(np.float64)
    if edge == "zero rows":
        assert (split[::3] == 0).all(), "a zero row must stay exactly zero"
    _judge(split, fma, ref)


def test_without_the_split_an_fp32_operand_keeps_the_exact_path(monkeypatch):
    """The representation is the profile's decision: with `fp32_split` withheld, an fp32 operand never reaches the
    unit (rounding it to fp16 would be wrong)."""
    from neurobrix.kernels.ops import _configs as C
    mu = {k: v for k, v in C.matrix_unit().items() if k != "fp32_split"}
    monkeypatch.setattr(C, "matrix_unit", lambda: mu)
    monkeypatch.setattr(W, "_matrix_unit", lambda: mu)
    calls = []
    real = W._gemm_m8n8k4
    monkeypatch.setattr(W, "_gemm_m8n8k4", lambda *a, **kw: (calls.append(1), real(*a, **kw))[1])
    rng = np.random.default_rng(3)
    a = rng.standard_normal((128, 64)).astype(np.float32) * 70000.0
    w = rng.standard_normal((64, 64)).astype(np.float16)
    got = W.mm(_nb(a), _nb(w)).numpy().astype(np.float64)
    assert not calls, "an fp32 operand took the fp16 matrix unit without the split"
    assert np.isfinite(got).all()
