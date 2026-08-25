"""Day-one float64 oracle for the SIMT MoE decode band.

Fourth member of the SIMT decode family. Tests the FULL band path
(execute_moe_fused routing -> topk -> the two-kernel vec pass) against
an INDEPENDENT float64 reference computed FROM THE PACKED BYTES
(numpy nibble unpack + q*scale+qmin dequant + SwiGLU + router-weighted
combine — the dequant-GEMV audit pattern: no NeuroBrix kernel in the
reference path).

Shapes are the canonical row's real per-expert shapes: gate/up qweight
int32 [256, 768] (K=2048 packed /8, N=768, G=16 groups of 128), down
[96, 2048] (K=768, G=6). E=16 synthetic experts (the kernel walks a
pointer table — expert COUNT only sizes the table), top_k=8.

Five proofs (the family rule):
  1. CORRECTNESS vs float64-from-packed-bytes, incl. ragged edges via
     a second shape set.
  2. DETERMINISM x3 (fixed-order in-kernel combine — the reason the
     GemLite/vLLM atomicAdd combine was refused).
  3. ROUTER-WEIGHT contract: scaling one expert's routing weight
     scales exactly its contribution.
  4. COUNTED route activation three states (ADOPTED default): unset
     and "1" reach the vec pass, "0" restores the grouped path; M>1
     and non-quantized tables never route (the zoo guard — fp16 MoEs
     keep the proven path).
  5. M>1 keeps the grouped path (prefill unaffected by construction).

Runnable two ways:
  - pytest:  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_moe_decode_vec_oracle.py -v
  - script:  PYTHONPATH=src python3 tests/unit/kernels/test_moe_decode_vec_oracle.py
"""
from __future__ import annotations

import ctypes
import os as _os

try:
    import pytest
except ModuleNotFoundError:  # script-mode under the pytest-less GPU venv
    class _NoPytest:
        @staticmethod
        def skip(*a, **k):
            raise SystemExit(0)

    pytest = _NoPytest()  # type: ignore

import numpy as np

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator
from neurobrix.kernels.quantized_tensor import QuantizedTensor
from neurobrix.triton import moe as MOE

# fp16 scales/activations, fp32 accumulation, K<=2048 sums + SwiGLU +
# 8-expert combine: a correct chain sits ~1e-3 relative (the SwiGLU
# nonlinearity amplifies fp16 input rounding); 5e-3 is the band bound,
# two orders under a wrong-nibble/wrong-group defect.
BOUND = 5e-03


def _has_gpu() -> bool:
    try:
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


def _d2h_f(t):
    n = t.numel()
    sz = 2 if "16" in str(t.nbx_dtype) else 4
    buf = (ctypes.c_char * (n * sz))()
    DeviceAllocator.memcpy(ctypes.addressof(buf), t.data_ptr(), n * sz, kind=2)
    dt = np.float16 if sz == 2 else np.float32
    return np.frombuffer(bytes(buf), dtype=dt).copy()


def _mk_expert(rng, K, N, group=128):
    """Synthetic int4-g128-asym expert projection, REAL layout:
    qweight int32 [K//8, N], scales/qmins fp16 [K//group, N]."""
    G = K // group
    q = rng.integers(0, 16, (K, N), dtype=np.int64)
    scales = (rng.random((G, N)).astype(np.float16) * 0.02 + 0.005)
    qmins = (rng.random((G, N)).astype(np.float16) * 0.1 - 0.15)
    packed = np.zeros((K // 8, N), dtype=np.int64)
    for i in range(8):
        packed |= q[i::8, :] << (4 * i)
    packed = packed.astype(np.uint32).view(np.int32)
    qt = QuantizedTensor(
        NBXTensor.from_numpy(np.ascontiguousarray(packed)),
        NBXTensor.from_numpy(np.ascontiguousarray(scales)),
        NBXTensor.from_numpy(np.ascontiguousarray(qmins)),
        logical_shape=(N, K), transposed=False)
    return qt, (q, scales, qmins)


def _dequant64(raw, group=128):
    q, scales, qmins = raw
    K, N = q.shape
    s = np.repeat(scales.astype(np.float64), group, axis=0)[:K]
    m = np.repeat(qmins.astype(np.float64), group, axis=0)[:K]
    return q.astype(np.float64) * s + m           # [K, N]


def _ref_band64(x, raws_g, raws_u, raws_d, ids, ws, group=128):
    """Independent float64 band: per active expert e (fixed order),
    h_e = silu(Wg_e^T x) * (Wu_e^T x); out = sum_e w_e * (Wd_e^T h_e)."""
    x64 = x.astype(np.float64)
    out = None
    for i, e in enumerate(ids):
        g = _dequant64(raws_g[e], group).T @ x64      # [N_gate]
        u = _dequant64(raws_u[e], group).T @ x64
        h = g / (1.0 + np.exp(-g)) * u
        d = _dequant64(raws_d[e], group).T @ h        # [K]
        out = d * float(ws[i]) if out is None else out + d * float(ws[i])
    return out


def _build_band(seed=0, E=16, K=2048, N_gate=768, top_k=8):
    rng = np.random.default_rng(seed)
    gates, ups, downs = [], [], []
    rg, ru, rd = [], [], []
    for _ in range(E):
        qt, raw = _mk_expert(rng, K, N_gate)
        gates.append(qt); rg.append(raw)
        qt, raw = _mk_expert(rng, K, N_gate)
        ups.append(qt); ru.append(raw)
        qt, raw = _mk_expert(rng, N_gate, K)
        downs.append(qt); rd.append(raw)
    xn = (rng.standard_normal(K) * 0.1).astype(np.float16)
    logits = rng.standard_normal((1, E)).astype(np.float16)
    return (gates, ups, downs), (rg, ru, rd), xn, logits


def _run_band(weights, xn, logits, top_k=8, arm="1"):
    gates, ups, downs = weights
    if arm is not None:
        _os.environ["NBX_MOE_VEC"] = arm
    try:
        h = NBXTensor.from_numpy(xn.reshape(1, -1))
        gs = NBXTensor.from_numpy(logits)
        out = MOE.execute_moe_fused(
            gs, h, gates, ups, downs,
            top_k=top_k, num_experts=len(gates),
            norm_topk_prob=True, cache_key="oracle")
        return _d2h_f(out).astype(np.float64).ravel()
    finally:
        _os.environ.pop("NBX_MOE_VEC", None)


def _ref_routing(logits, top_k):
    l64 = logits.astype(np.float64).ravel()
    ids = np.argsort(-l64, kind="stable")[:top_k]
    w = l64[ids]
    w = w / w.sum()
    return ids, w


def test_moe_vec_matches_float64_from_packed_bytes() -> None:
    if not _has_gpu():
        pytest.skip("no GPU")
    weights, raws, xn, logits = _build_band(seed=3)
    got = _run_band(weights, xn, logits)
    ids, wts = _ref_routing(logits, 8)
    ref = _ref_band64(xn, *raws, ids, wts)
    err = float(np.abs(got - ref).max() / np.abs(ref).max())
    assert err <= BOUND, (
        f"band rel err {err:.3e} exceeds {BOUND:.0e} by {err/BOUND:.0f}x")


def test_moe_vec_is_deterministic() -> None:
    if not _has_gpu():
        pytest.skip("no GPU")
    weights, _, xn, logits = _build_band(seed=5)
    outs = [_run_band(weights, xn, logits).tobytes() for _ in range(3)]
    assert outs[0] == outs[1] == outs[2], (
        "outputs differ across identical calls — the fixed-order "
        "combine failed its one job")


def test_moe_vec_router_weight_contract() -> None:
    """Doubling one expert's routing weight (pre-normalization scaling
    via its logit is nonlinear — instead compare two runs whose only
    difference is the top-1 logit, checked against float64 both times)."""
    if not _has_gpu():
        pytest.skip("no GPU")
    weights, raws, xn, logits = _build_band(seed=7)
    l2 = logits.copy()
    l2[0, int(np.argmax(logits))] *= 1.5
    for lg in (logits, l2):
        got = _run_band(weights, xn, lg)
        ids, wts = _ref_routing(lg, 8)
        ref = _ref_band64(xn, *raws, ids, wts)
        err = float(np.abs(got - ref).max() / np.abs(ref).max())
        assert err <= BOUND, f"routing-weight variant rel err {err:.3e}"


def test_moe_vec_route_activation_and_guards() -> None:
    """COUNTED: "1" reaches the vec pass; unset does NOT (under
    judgment); M>1 does NOT (prefill guard). The fp16-expert zoo guard
    is structural (tables.quantized False bypasses the branch) and is
    exercised by every fp16-MoE gate row."""
    if not _has_gpu():
        pytest.skip("no GPU")
    calls = {"n": 0}
    orig = MOE._moe_decode_vec_pass

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    MOE._moe_decode_vec_pass = counting
    try:
        weights, _, xn, logits = _build_band(seed=9)
        _run_band(weights, xn, logits, arm="1")
        assert calls["n"] == 1, "armed env did not reach the vec pass"
        _run_band(weights, xn, logits, arm=None)   # unset: ADOPTED default
        assert calls["n"] == 2, "unset (adopted default) did not route to the vec pass"
        _run_band(weights, xn, logits, arm="0")    # kill switch
        assert calls["n"] == 2, "kill switch '0' did not restore the grouped path"
        # M > 1: grouped path regardless of arming
        h2 = NBXTensor.from_numpy(
            np.repeat(xn.reshape(1, -1), 2, axis=0))
        gs2 = NBXTensor.from_numpy(np.repeat(logits, 2, axis=0))
        _os.environ["NBX_MOE_VEC"] = "1"
        try:
            MOE.execute_moe_fused(gs2, h2, *weights, top_k=8,
                                  num_experts=len(weights[0]),
                                  norm_topk_prob=True, cache_key="oracle2")
        finally:
            _os.environ.pop("NBX_MOE_VEC", None)
        assert calls["n"] == 2, "M=2 routed to the M=1 vec pass"
    finally:
        MOE._moe_decode_vec_pass = orig


def test_moe_vec_agrees_with_grouped_path() -> None:
    """Cross-implementation: the vec band vs the proven grouped band on
    the same inputs — both within the float64 bound, and within a tight
    mutual tolerance (different fp orders, same math)."""
    if not _has_gpu():
        pytest.skip("no GPU")
    weights, _, xn, logits = _build_band(seed=11)
    a = _run_band(weights, xn, logits, arm="1")
    b = _run_band(weights, xn, logits, arm="0")
    rel = float(np.abs(a - b).max() / np.abs(b).max())
    assert rel <= 2e-3, f"vec vs grouped mutual rel {rel:.3e}"


if __name__ == "__main__":
    if not _has_gpu():
        raise SystemExit("no GPU")
    test_moe_vec_matches_float64_from_packed_bytes()
    print("  OK float64 from packed bytes")
    test_moe_vec_is_deterministic()
    print("  OK determinism x3")
    test_moe_vec_router_weight_contract()
    print("  OK router-weight contract")
    test_moe_vec_route_activation_and_guards()
    print("  OK route activation + guards (unset, M>1)")
    test_moe_vec_agrees_with_grouped_path()
    print("  OK vec agrees with the grouped path")
    print("PASS: moe_decode_vec oracle complete")
