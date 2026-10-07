"""Unit test — `mm`, `addmm`, `bmm` and `baddbmm` on the m8n8k4 matrix unit hold the GEMM contract of the tl.dot kernels.

Where the hardware profile declares `matrix_unit` with `mm` tiles (volta.yml), a GEMM whose two operands are the
unit's operand dtype in memory runs `kernels/ops/matmul_m8n8k4.py`. Each clause of the contract is proven against
an fp64 oracle computed in-test with the route asserted TAKEN: M/N/K tails, a pre-transposed weight walked by its
strides, the bias with alpha and beta, the output dtype, and the fused epilogues byte-identical to the unfused
pair (`_apply_matmul_epilogue` — the decode path's own standalone wrappers). An fp32 operand is never rounded
to fp16: it reaches the unit only as the profile's hi + lo split. Skipped where the profile declares no matrix unit.

  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_gemm_m8n8k4.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels import wrappers as W
from neurobrix.kernels.nbx_tensor import DeviceAllocator, NBXTensor


def _unit_declared() -> bool:
    try:
        DeviceAllocator.set_device(0)
        return bool((W._matrix_unit() or {}).get("mm"))
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _unit_declared(), reason="the hardware profile declares no matrix_unit.mm")


@pytest.fixture(autouse=True, scope="module")
def _profile():
    from neurobrix.kernels import autotune_certify as Z
    Z._bind_hardware_profile()          # the wrappers' dtype policy of this card (an fp32 store without native bf16)


@pytest.fixture
def taken(monkeypatch):
    calls = []
    real = W._gemm_m8n8k4
    monkeypatch.setattr(W, "_gemm_m8n8k4", lambda *a, **kw: (calls.append(a[0].shape), real(*a, **kw))[1])
    return calls


def _nb(x):
    return NBXTensor.from_numpy(np.ascontiguousarray(x)).to("cuda:0")


def _rel(got, ref):
    return np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30)


def _bound(out):
    """The output's own rounding: an fp16 C carries one half-ulp per element, the fp32 sums stay far below it; an
    fp32 C carries the fp32 accumulation over K (1e-5 of the largest value at these K)."""
    return max(float(np.finfo(out.dtype).eps), 1e-5)


CASES = [
    # M, N, K, weight pre-transposed
    (512, 512, 512, False),
    (777, 333, 100, False),     # every tail
    (300, 1152, 1152, True),    # a linear's weight, read in place through its strides
    (65, 70, 36, True),
    (4096, 128, 64, False),
]


@pytest.mark.parametrize("M,N,K,bt", CASES)
def test_mm_matches_fp64(M, N, K, bt, taken):
    rng = np.random.default_rng(M + N + K)
    a = (rng.standard_normal((M, K)) * 0.5).astype(np.float16)
    w = (rng.standard_normal((K, N)) * 0.5).astype(np.float16)
    b = _nb(w.T.copy()).t() if bt else _nb(w)
    assert (b.stride(0) == 1) == bt
    out = W.mm(_nb(a), b).numpy()
    got = out.astype(np.float64)
    assert taken, "the m8n8k4 route was not taken"
    ref = a.astype(np.float64) @ w.astype(np.float64)
    assert np.isfinite(got).all()
    assert _rel(got, ref) < _bound(out), f"rel {_rel(got, ref):.2e} dtype {out.dtype}"


@pytest.mark.parametrize("bias_dtype", [np.float16, np.float32])
def test_addmm_bias_alpha_beta(bias_dtype, taken):
    rng = np.random.default_rng(7)
    M, N, K = 200, 136, 72
    a = (rng.standard_normal((M, K)) * 0.5).astype(np.float16)
    w = (rng.standard_normal((N, K)) * 0.5).astype(np.float16)
    bias = rng.standard_normal(N).astype(bias_dtype)
    out = W.addmm(_nb(bias), _nb(a), _nb(w).t(), beta=0.5, alpha=2.0).numpy()
    got = out.astype(np.float64)
    assert taken, "the m8n8k4 route was not taken"
    ref = 0.5 * bias.astype(np.float64) + 2.0 * (a.astype(np.float64) @ w.astype(np.float64).T)
    assert _rel(got, ref) < _bound(out), f"rel {_rel(got, ref):.2e} dtype {out.dtype}"


@pytest.mark.parametrize("code", [1, 2, 3])
@pytest.mark.parametrize("with_bias", [False, True])
def test_fused_epilogue_is_the_unfused_pair(code, with_bias, taken):
    rng = np.random.default_rng(code)
    M, N, K = 160, 96, 64
    a, w = _nb((rng.standard_normal((M, K))).astype(np.float16)), _nb((rng.standard_normal((N, K))).astype(np.float16)).t()
    bias = _nb(rng.standard_normal(N).astype(np.float16))
    run = (lambda e: W.addmm(bias, a, w, _epilogue=e)) if with_bias else (lambda e: W.mm(a, w, _epilogue=e))
    fused = run(code).numpy()
    unfused = W._apply_matmul_epilogue(run(0), code).numpy()
    assert taken, "the m8n8k4 route was not taken"
    assert fused.dtype == unfused.dtype
    assert np.array_equal(fused.view(np.uint8), unfused.view(np.uint8)), \
        f"max|diff| {np.abs(fused.astype(np.float64) - unfused).max():.2e}"


@pytest.mark.parametrize("Z,M,N,K,bt", [(3, 300, 136, 72, False), (5, 77, 64, 96, True), (2, 1024, 1024, 128, False)])
def test_bmm_matches_fp64(Z, M, N, K, bt, taken):
    rng = np.random.default_rng(Z * M + K)
    a = (rng.standard_normal((Z, M, K)) * 0.5).astype(np.float16)
    w = (rng.standard_normal((Z, K, N)) * 0.5).astype(np.float16)
    b = _nb(np.swapaxes(w, 1, 2).copy()).transpose(1, 2) if bt else _nb(w)      # K^T of an attention, by strides
    out = W.bmm(_nb(a), b, allow_strided_b=bt).numpy()
    assert taken, "the m8n8k4 route was not taken"
    ref = np.einsum("zmk,zkn->zmn", a.astype(np.float64), w.astype(np.float64))
    assert out.dtype == np.float32                                   # the store dtype is the batched kernel's
    assert _rel(out.astype(np.float64), ref) < _bound(out), f"rel {_rel(out.astype(np.float64), ref):.2e}"


@pytest.mark.parametrize("bias_shape", [(200, 136), (3, 1, 136), (3, 200, 136), (1, 136)])
def test_baddbmm_bias_broadcasts(bias_shape, taken):
    rng = np.random.default_rng(len(bias_shape) + bias_shape[-2])
    Z, M, N, K = 3, 200, 136, 72
    a = (rng.standard_normal((Z, M, K)) * 0.5).astype(np.float16)
    w = (rng.standard_normal((Z, K, N)) * 0.5).astype(np.float16)
    bias = rng.standard_normal(bias_shape).astype(np.float16)
    out = W.baddbmm_wrapper(_nb(bias), _nb(a), _nb(w), beta=0.5, alpha=2.0).numpy()
    assert taken, "the m8n8k4 route was not taken"
    ref = 0.5 * bias.astype(np.float64) + 2.0 * np.einsum("zmk,zkn->zmn", a.astype(np.float64), w.astype(np.float64))
    assert _rel(out.astype(np.float64), ref) < _bound(out), f"rel {_rel(out.astype(np.float64), ref):.2e}"


def test_an_fp32_operand_is_never_rounded_to_the_operand_dtype(monkeypatch):
    """An fp32 operand reaches the unit only as the split (hi + lo, tests/unit/kernels/test_gemm_fp32_split.py),
    never as its fp16 rounding."""
    tiles = []
    real = W._gemm_m8n8k4
    monkeypatch.setattr(W, "_gemm_m8n8k4", lambda *a, **kw: (tiles.append(a[8]), real(*a, **kw))[1])
    rng = np.random.default_rng(3)
    a = rng.standard_normal((128, 64)).astype(np.float32) * 70000.0     # beyond fp16: rounding it would be wrong
    w = rng.standard_normal((64, 64)).astype(np.float16)
    got = W.mm(_nb(a), _nb(w)).numpy().astype(np.float64)
    assert all(t["split_a"] and not t["split_b"] for t in tiles), tiles
    ref = a.astype(np.float64) @ w.astype(np.float64)
    assert _rel(got, ref) < 1e-5, f"rel {_rel(got, ref):.2e}"


def test_the_census_derives_no_key_where_the_unit_runs():
    """The recorded key IS the launcher's key: a GEMM the matrix unit takes has no autotune key, so the derived
    census (launch_keys, the same predicate as the wrappers) lists none, fp32 operands (split) included; a GEMM
    below the unit's smallest tile keeps its key."""
    from neurobrix.kernels import launch_keys as LK
    from neurobrix.kernels.nbx_tensor import NBXDtype
    h, f = NBXDtype.float16, NBXDtype.float32
    assert LK.mm_launches(512, 512, 512, h, h, False) == []
    assert LK.addmm_launches(512, 512, 512, h, h, h, False) == []
    assert LK.addmm_launches(512, 512, 512, h, h, h, False, bias_row=False) != []    # a full bias keeps the kernel
    assert LK.bmm_launches(512, 512, 512, h, h, False) == []
    assert LK.baddbmm_launches(512, 512, 512, h, h, h) == []
    assert LK.bmm_launches(1, 512, 512, h, h, False) != []                           # decode keeps the batched kernel
    assert LK.mm_launches(512, 512, 512, f, h, False) == []                          # fp32: the hi + lo split
    assert LK.baddbmm_launches(512, 512, 512, f, f, f) == []
    assert LK.mm_launches(16, 512, 512, f, h, False) != []                          # below the unit's smallest tile
