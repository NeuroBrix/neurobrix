"""P-NONDET-LONG-ROW pinning tests — deterministic long-context prefill.

The measured cause chain (2026-08-23): the long row's prefill (T=4,164,
D=128 pow2) has an fp32 scores tensor 3% OVER the Volta math budget →
routed to the flash kernel → the exact D>=128 pad-by-one detour lands
it in the masked-load specialisation → the DOCUMENTED Volta race:
kernel-level 3 distinct outputs in 5 identical calls at that shape;
end-to-end the first replayed-step logits differed on EVERY pair of
runs, and one greedy near-tie (~token 10) made the row's output
bimodal (the stable sha pair). Forcing math: 4/4 runs byte-identical
at every step. Fix: chunked math for over-budget pow2 shapes, bounded
by `memory.sdpa_math_max_chunks` (vendor yml).

Three pins:
  1. DETERMINISM at the exact trigger shape through the chunked route
     — three calls, identical bytes. FAILS on the pre-fix flash route
     (measured 3 distinct in 5).
  2. CORRECTNESS: chunked output vs the independent float64 reference.
  3. ROUTING + ACTIVATION: over-budget pow2 routes to
     _math_attention_chunked when the ceiling allows; keeps flash when
     the chunk count exceeds it (the registered video-class residual);
     and the route is proven by a call counter, not assumed.

The budget helpers are patched in-test (a bare test process has no
hardware profile — the vacuous-equivalence lesson: pin the route).

Runnable two ways:
  - pytest:  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_prefill_determinism.py -v
  - script:  PYTHONPATH=src python3 tests/unit/kernels/test_prefill_determinism.py
"""
from __future__ import annotations

import ctypes

try:
    import pytest
except ModuleNotFoundError:  # script-mode under the pytest-less GPU venv
    class _NoPytest:
        @staticmethod
        def skip(*a, **k):
            raise SystemExit(0)

    pytest = _NoPytest()  # type: ignore

import numpy as np

from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator
from neurobrix.kernels import wrappers as W

BOUND = 2e-03  # same derived bound as the decode oracles


def _has_gpu() -> bool:
    try:
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


def _d2h(t) -> bytes:
    buf = (ctypes.c_char * t._nbytes)()
    DeviceAllocator.memcpy(ctypes.addressof(buf), t.data_ptr(),
                           t._nbytes, 2)
    return bytes(buf)


class _pinned_route:
    """Patch the two budget helpers so the chunked route is decided by
    THIS test, not by the machine's profile."""

    def __init__(self, budget, max_chunks):
        self.budget, self.max_chunks = budget, max_chunks

    def __enter__(self):
        self._b = W._sdpa_math_scores_budget_bytes
        self._c = W._sdpa_math_max_chunks
        W._sdpa_math_scores_budget_bytes = lambda: self.budget
        W._sdpa_math_max_chunks = lambda: self.max_chunks
        return self

    def __exit__(self, *a):
        W._sdpa_math_scores_budget_bytes = self._b
        W._sdpa_math_max_chunks = self._c


def _mk(B, H, H_kv, T, D, seed=7):
    rng = np.random.default_rng(seed)
    q = NBXTensor.from_numpy(
        (rng.standard_normal((B, H, T, D)) * 0.1).astype(np.float16))
    k = NBXTensor.from_numpy(
        (rng.standard_normal((B, H_kv, T, D)) * 0.1).astype(np.float16))
    v = NBXTensor.from_numpy(
        (rng.standard_normal((B, H_kv, T, D)) * 0.1).astype(np.float16))
    return q, k, v


def test_long_prefill_is_deterministic_through_the_chunked_route() -> None:
    """The exact trigger: pow2 D, causal, scores just over budget.
    Three identical calls, identical bytes. The pre-fix flash route
    measured 3 distinct outputs in 5 at this shape class."""
    if not _has_gpu():
        pytest.skip("no GPU")
    B, H, H_kv, T, D = 1, 32, 8, 4164, 128
    q, k, v = _mk(B, H, H_kv, T, D)
    calls = {"n": 0}
    orig = W._math_attention_chunked

    def counting(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)

    W._math_attention_chunked = counting
    try:
        with _pinned_route(budget=2147483648, max_chunks=16):
            outs = [_d2h(W.scaled_dot_product_attention_wrapper(
                q, k, v, attn_mask=None, is_causal=True,
                scale=1.0 / np.sqrt(D))) for _ in range(3)]
    finally:
        W._math_attention_chunked = orig
    assert calls["n"] == 3, (
        f"chunked route fired {calls['n']} times, wanted 3 — the "
        f"trigger shape did not route as pinned (vacuous determinism)")
    assert outs[0] == outs[1] == outs[2], (
        "long-prefill outputs differ across identical calls — the "
        "P-NONDET-LONG-ROW race is back")


def test_chunked_matches_float64() -> None:
    """Chunked math vs the independent float64 reference (causal)."""
    if not _has_gpu():
        pytest.skip("no GPU")
    B, H, H_kv, T, D = 1, 8, 2, 512, 64
    rng = np.random.default_rng(3)
    qn = (rng.standard_normal((B, H, T, D)) * 0.1).astype(np.float16)
    kn = (rng.standard_normal((B, H_kv, T, D)) * 0.1).astype(np.float16)
    vn = (rng.standard_normal((B, H_kv, T, D)) * 0.1).astype(np.float16)
    q, k, v = (NBXTensor.from_numpy(x) for x in (qn, kn, vn))
    out = W._math_attention_chunked(q, k, v, None, True,
                                    1.0 / np.sqrt(D), chunk_rows=128)
    got = np.frombuffer(_d2h(out), dtype=np.float16).astype(
        np.float64).reshape(B, H, T, D)

    g = H // H_kv
    K = np.repeat(kn.astype(np.float64), g, axis=1)
    V = np.repeat(vn.astype(np.float64), g, axis=1)
    s = np.einsum("bhqd,bhkd->bhqk", qn.astype(np.float64), K) / np.sqrt(D)
    tri = np.triu(np.ones((T, T)), k=1).astype(bool)
    s[:, :, tri] = -np.inf
    m = s.max(-1, keepdims=True)
    e = np.exp(s - m)
    ref = np.einsum("bhqk,bhkd->bhqd", e / e.sum(-1, keepdims=True), V)
    err = float(np.abs(got - ref).max() / np.abs(ref).max())
    assert err <= BOUND, f"chunked math rel err {err:.3e} > {BOUND:.0e}"


def test_video_scale_keeps_its_path() -> None:
    """A shape whose chunk count exceeds the ceiling must NOT take the
    chunked route (the registered residual class keeps its behavior)."""
    if not _has_gpu():
        pytest.skip("no GPU")
    calls = {"n": 0}
    orig = W._math_attention_chunked

    def counting(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)

    W._math_attention_chunked = counting
    try:
        # tiny budget => chunk_rows 128, T=4096 => 32 chunks > ceiling 16
        with _pinned_route(budget=128 * 1 * 4 * 4096 * 4, max_chunks=16):
            B, H, H_kv, T, D = 1, 4, 4, 4096, 64
            q, k, v = _mk(B, H, H_kv, T, D, seed=9)
            W.scaled_dot_product_attention_wrapper(
                q, k, v, attn_mask=None, is_causal=True,
                scale=1.0 / np.sqrt(D))
    finally:
        W._math_attention_chunked = orig
    assert calls["n"] == 0, (
        "over-ceiling shape took the chunked route — the video-class "
        "cost bound is not being honored")


if __name__ == "__main__":
    if not _has_gpu():
        raise SystemExit("no GPU")
    test_long_prefill_is_deterministic_through_the_chunked_route()
    print("PASS: long-prefill deterministic x3 through the chunked route")
    test_chunked_matches_float64()
    print("PASS: chunked math matches float64")
    test_video_scale_keeps_its_path()
    print("PASS: over-ceiling shapes keep their path")
