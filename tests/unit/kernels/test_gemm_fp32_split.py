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

from nbxtensor.ops import wrappers as W
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.tensor import NBXTensor


def _no_gpu_reason():
    """Why these cells cannot run here, or None. Only the ABSENCE of a device or of the profile's split rows skips: a
    broken kernel module, a refused profile or a failing allocator raises at collection, never reads as a skip."""
    from nbxtensor.backend.runtime import _gpu_runtime
    try:
        _gpu_runtime()
    except (OSError, RuntimeError) as exc:           # no CUDA/HIP runtime library on this host
        return f"no GPU runtime: {exc}"
    if DeviceAllocator.device_count() < 1:           # a runtime with no visible device (CUDA_VISIBLE_DEVICES="")
        return "the process sees no device"
    DeviceAllocator.set_device(0)
    if not ((W._matrix_unit() or {}).get("fp32_split") or {}).get("mm"):
        return "the hardware profile declares no matrix_unit.fp32_split"
    return None


_SKIP = _no_gpu_reason()
pytestmark = pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")

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
    from nbxtensor.ops import _configs as C
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


def test_a_bfloat16_operand_through_the_split_rounds_to_the_oracle(monkeypatch):
    """`fp32_split.operands` lists bfloat16: a bf16 x bf16 GEMM takes the split (flags True, True) and its bf16 output
    is the correctly rounded fp64 product on all but the elements within the inside's fp32-level error of a rounding
    boundary: at most 1 % of them, each by one bf16 ulp of the binade the oracle or the result lies in, plus the
    inside's fp32-level error against the uncancelled sum sum_k |a_ik w_kj| (a dot product's error scales with that
    sum, not with its result; first measurement 2026-10-12: 0.010 % off, max exactly 2.00 'ulp' — the gate's ulp
    constant was 2^-8 where bfloat16's spacing is 2^-7, so a one-ulp element read as two). An fp16-level inside would put half the elements off. The FMA path on the
    same inputs is held to the same bound."""
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.dtype.convert import bf16_bits_to_f32 as bf16_carrier_to_float32, f32_to_bf16_bits as float32_to_bf16_bits

    def bf16(x):
        bits = float32_to_bf16_bits(np.ascontiguousarray(x, dtype=np.float32))
        return bf16_carrier_to_float32(np.asarray(bits, dtype=np.uint16).view("V2")).astype(np.float64)
    rng = np.random.default_rng(16)
    M, K, N = 512, 1024, 256
    a = bf16(rng.standard_normal((M, K)))
    w = bf16(rng.standard_normal((K, N)) / np.sqrt(K))
    A, Wb = _nb(a.astype(np.float32)).to(NBXDtype.bfloat16), _nb(w.astype(np.float32)).to(NBXDtype.bfloat16)
    ref = a @ w
    uncancelled = np.abs(a) @ np.abs(w)
    split, flags, fma = _both(lambda: W.mm(A, Wb).to(NBXDtype.float32), monkeypatch)
    assert flags == (True, True)
    for name, got in (("split", split), ("FMA", fma)):
        off = np.abs(got - bf16(ref))
        ulp = 2.0 ** -7 * 2.0 ** np.floor(np.log2(np.maximum(np.maximum(np.abs(ref), np.abs(got)), 1e-300)))   # bf16: 7 stored bits
        bound = ulp + _SLACK * uncancelled
        frac = float((off > 0).mean())
        print(f"\n  bf16 {name}: {frac * 100:.3f} % off the rounded oracle, max {float((off / ulp).max()):.2f} ulp of the "
              f"binade, max {float((off / bound).max()):.2f} of the bound")
        assert np.isfinite(got).all()
        assert (off <= bound * (1 + 1e-9)).all(), \
            f"{name}: an element further from the rounded oracle than one ulp of its binade plus the inside's error on the uncancelled sum"
        assert frac <= 0.01, f"{name}: {frac * 100:.2f} % of the elements off the correctly rounded oracle"


def test_without_the_split_an_fp32_operand_keeps_the_exact_path(monkeypatch):
    """The representation is the profile's decision: with `fp32_split` withheld, an fp32 operand never reaches the
    unit (rounding it to fp16 would be wrong)."""
    from nbxtensor.ops import _configs as C
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
