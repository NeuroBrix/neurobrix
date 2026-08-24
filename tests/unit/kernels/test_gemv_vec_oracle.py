"""Day-one float64 oracle for the SIMT gemv_vec kernel.

Third member of the SIMT decode family (decode_attn_vec, adopted;
this one under judgment). Shapes are the REAL five call sites of the
canonical row's decode, censused via NBX_MM_SHAPES on 2026-08-24:
K/V proj (512, 2048) x2/layer, Q proj (4096, 2048), O proj
(2048, 4096), lm_head (151936, 2048) — 193 calls/token, ~2.43
GB/token of fp16 weight reads (lm_head alone 26%).

Five proofs (the family rule): float64 correctness on the real
shapes + ragged K; determinism x3; masked tail (K not a BLOCK_K
multiple contributes exactly its elements); COUNTED route activation
(the env actually reaches the kernel; off by default during
judgment); mixed-dtype contract (fp16 weights x fp32 activation — the
Volta AMP shape).

Runnable two ways:
  - pytest:  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_gemv_vec_oracle.py -v
  - script:  PYTHONPATH=src python3 tests/unit/kernels/test_gemv_vec_oracle.py
"""
from __future__ import annotations

import ctypes
import os as _os

try:
    import pytest
except ModuleNotFoundError:  # script-mode under the pytest-less GPU venv
    class _NoPytest:
        class mark:
            @staticmethod
            def parametrize(*a, **k):
                return lambda fn: fn

        @staticmethod
        def skip(*a, **k):
            raise SystemExit(0)

    pytest = _NoPytest()  # type: ignore

import numpy as np

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator
from neurobrix.kernels import wrappers as W

# fp16 weights, fp32 activation, fp32 accumulation over K<=4096 terms:
# a correct kernel sits at ~1e-4 relative; 2e-3 is the family bound.
BOUND = 2e-03

_NP = {NBXDtype.float16: (np.float16, 2), NBXDtype.float32: (np.float32, 4)}


def _has_gpu() -> bool:
    try:
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


def _d2h(t):
    dt, sz = _NP[t._dtype]
    buf = (ctypes.c_char * (t.numel() * sz))()
    DeviceAllocator.memcpy(ctypes.addressof(buf), t.data_ptr(),
                           t.numel() * sz, kind=2)
    return np.frombuffer(bytes(buf), dtype=dt).copy()


def _mk(N, K, seed=0):
    rng = np.random.default_rng(seed)
    mat = NBXTensor.from_numpy(
        (rng.standard_normal((N, K)) * 0.05).astype(np.float16))
    vec = NBXTensor.from_numpy(
        (rng.standard_normal(K) * 0.05).astype(np.float32))
    return mat, vec


def _run_vec(mat, vec):
    _os.environ["NBX_MV_VEC"] = "1"
    try:
        return W.mv_wrapper(mat, vec)
    finally:
        _os.environ.pop("NBX_MV_VEC", None)


# The five real call sites + ragged-K edges.
_SHAPES = [
    (512, 2048),       # K/V projection
    (4096, 2048),      # Q projection
    (2048, 4096),      # O projection
    (151936, 2048),    # lm_head (vocab)
    (2048, 4096 - 3),  # ragged K (not a BLOCK_K multiple)
    (100, 130),        # tiny + ragged both dims
]


@pytest.mark.parametrize("N,K", _SHAPES)
def test_gemv_vec_matches_float64(N, K) -> None:
    if not _has_gpu():
        pytest.skip("no GPU")
    mat, vec = _mk(N, K)
    out = _run_vec(mat, vec)
    got = _d2h(out).astype(np.float64)
    mn = _d2h(mat).astype(np.float64).reshape(N, K)
    vn = _d2h(vec).astype(np.float64)
    ref = mn @ vn
    err = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9))
    assert err <= BOUND, (
        f"N={N} K={K}: rel err {err:.3e} exceeds {BOUND:.0e} "
        f"by {err/BOUND:.0f}x")


def test_gemv_vec_is_deterministic() -> None:
    """Three calls, identical bytes — replay-engine requirement (all of
    K inside one program, fixed chunk order, no atomics)."""
    if not _has_gpu():
        pytest.skip("no GPU")
    mat, vec = _mk(4096, 2048, seed=5)
    outs = [_d2h(_run_vec(mat, vec)).tobytes() for _ in range(3)]
    assert outs[0] == outs[1] == outs[2], (
        "outputs differ across identical calls — disqualified for the "
        "replay engine")


def test_gemv_vec_masked_tail_inert() -> None:
    """With ragged K, bytes BEYOND K in the same rows must not leak:
    compare against a padded matrix whose tail columns are garbage."""
    if not _has_gpu():
        pytest.skip("no GPU")
    N, K = 256, 1000  # not a BLOCK_K multiple
    rng = np.random.default_rng(11)
    base = (rng.standard_normal((N, 1024)) * 0.05).astype(np.float16)
    a1 = base.copy()
    a2 = base.copy()
    a2[:, K:] = 999.0  # garbage beyond K
    vn = (rng.standard_normal(K) * 0.05).astype(np.float32)
    m1 = NBXTensor.from_numpy(np.ascontiguousarray(a1[:, :K]))
    m2 = NBXTensor.from_numpy(np.ascontiguousarray(a2[:, :K]))
    v = NBXTensor.from_numpy(vn)
    o1 = _d2h(_run_vec(m1, v)).tobytes()
    o2 = _d2h(_run_vec(m2, v)).tobytes()
    assert o1 == o2, "tail bytes leaked into the reduction"


def test_gemv_vec_route_activation_counted() -> None:
    """COUNTED three states (ADOPTED default, 2026-08-24 verdict):
    default routes to gemv_vec; NBX_MV_VEC=0 restores the incumbent."""
    if not _has_gpu():
        pytest.skip("no GPU")
    from neurobrix.kernels.ops import gemv_vec as G
    calls = {"n": 0}
    orig = G.gemv_vec_kernel

    class Counting:
        def __getitem__(self, grid):
            def launch(*a, **k):
                calls["n"] += 1
                return orig[grid](*a, **k)
            return launch

    G.gemv_vec_kernel = Counting()
    try:
        mat, vec = _mk(512, 2048, seed=3)
        _run_vec(mat, vec)          # "1": armed
        assert calls["n"] == 1, "armed env did not reach gemv_vec_kernel"
        W.mv_wrapper(mat, vec)      # default (unset): ADOPTED -> vec
        assert calls["n"] == 2, "default did not route to gemv_vec"
        _os.environ["NBX_MV_VEC"] = "0"
        try:
            W.mv_wrapper(mat, vec)  # kill switch -> incumbent
        finally:
            _os.environ.pop("NBX_MV_VEC", None)
        assert calls["n"] == 2, "kill switch '0' did not restore the incumbent"
    finally:
        G.gemv_vec_kernel = orig


def test_gemv_vec_fp16_weight_fp32_activation() -> None:
    """The exact Volta AMP dtype pair of the decode route: weights read
    as fp16 (the bandwidth carrier), activation fp32."""
    if not _has_gpu():
        pytest.skip("no GPU")
    mat, vec = _mk(2048, 4096, seed=7)
    assert mat.nbx_dtype == NBXDtype.float16
    assert vec.nbx_dtype == NBXDtype.float32
    out = _run_vec(mat, vec)
    got = _d2h(out).astype(np.float64)
    ref = _d2h(mat).astype(np.float64).reshape(2048, 4096) @ \
        _d2h(vec).astype(np.float64)
    err = float(np.abs(got - ref).max() / np.abs(ref).max())
    assert err <= BOUND


if __name__ == "__main__":
    if not _has_gpu():
        raise SystemExit("no GPU")
    for shp in _SHAPES:
        mat, vec = _mk(*shp)
        out = _run_vec(mat, vec)
        got = _d2h(out).astype(np.float64)
        ref = _d2h(mat).astype(np.float64).reshape(shp) @ \
            _d2h(vec).astype(np.float64)
        err = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9))
        print(f"  {'OK ' if err <= BOUND else 'FAIL'} {shp}: {err:.3e}")
        assert err <= BOUND, shp
    test_gemv_vec_is_deterministic()
    print("  OK determinism x3")
    test_gemv_vec_masked_tail_inert()
    print("  OK masked tail inert")
    test_gemv_vec_route_activation_counted()
    print("  OK route activation counted")
    test_gemv_vec_fp16_weight_fp32_activation()
    print("  OK fp16 x fp32 contract")
    print("PASS: gemv_vec oracle complete")
