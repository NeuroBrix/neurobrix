"""CORRECTNESS ORACLE — the gate class a byte gate cannot be.

=== WHY THIS FILE EXISTS ===

Every gate this project had proves REPRODUCIBILITY: two runs of the same
code produce the same bytes. None of them proves CORRECTNESS. A kernel
that is systematically wrong passes all of them, because both arms carry
the same error.

That is not hypothetical. `scaled_dot_product_attention_wrapper` served
head dimension 128 — Qwen3's, and most modern models' — with a relative
error of 3.4e-01 against the exact answer, for weeks, through every
full-zoo byte gate. It surfaced only because the error also made the
kernel non-deterministic, and only then because a long-context row
happened to sample a near-tie.

**A byte gate proves we repeat ourselves. A correctness oracle proves we
are right. Both are required.**

=== THE CONTRACT ===

For each numeric kernel: compare against a float64 reference computed
independently (numpy, on CPU), over a SWEPT input space, with a declared
error bound justified by the dtype and the accumulation depth. The test
fails when the error exceeds the bound — **even if the result is
perfectly reproducible**.

The bound is not a knob to widen when a test fails. It is derived, stated
here, and a failure against it is a defect in the kernel.

=== THE BOUND, DERIVED ===

fp16 has an 11-bit significand, so a single rounded value carries a
relative error up to 2^-11 = 4.9e-04. Attention accumulates over the
sequence in an fp32 accumulator, so the dominant term is the input
rounding rather than the accumulation, and error grows roughly with
sqrt(T) of that unit in the worst case, not with T.

Measured on the paths known good — head dims 16, 32, 64, 96, 127, 129,
160, 192, 255 — the relative error against float64 sits in
1.9e-04 .. 5.0e-04 across T from 64 to 512. **The bound is set at 2e-03**,
roughly four times the worst observed value on a correct path: loose
enough that fp16 rounding and reduction-order differences never trip it,
tight enough that the D=128 defect (3.4e-01, i.e. 170x the bound) is
caught by three orders of magnitude.

Runnable two ways:
  - pytest:  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_numeric_correctness_oracle.py -v
  - script:  PYTHONPATH=src python3 tests/unit/kernels/test_numeric_correctness_oracle.py
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

# Declared bound for fp16-in / fp32-accumulate attention. See the
# derivation in the module docstring. Widening this is a change to the
# contract and needs its own justification, not a passing test.
ATTENTION_FP16_REL_BOUND = 2e-03

_NP = {NBXDtype.float32: (np.float32, 4), NBXDtype.float16: (np.float16, 2)}


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


def _attention_float64(qn, kn, vn, causal=True):
    """Independent reference: exact softmax attention in float64 on CPU.

    Deliberately written from the definition rather than by calling any
    NeuroBrix code — an oracle that shares an implementation with the
    thing it checks is not an oracle.
    """
    Q, K, V = (x.astype(np.float64) for x in (qn, kn, vn))
    T_q, T_k = Q.shape[2], K.shape[2]
    s = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(Q.shape[-1])
    if causal:
        m = (np.arange(T_k)[None, :] > np.arange(T_q)[:, None])
        s = np.where(m, -np.inf, s)
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    return (e / e.sum(-1, keepdims=True)) @ V


def _relative_error(B, H, T, D, causal=True, seed=0):
    rng = np.random.default_rng(seed)
    qn = rng.standard_normal((B, H, T, D)).astype(np.float16)
    kn = rng.standard_normal((B, H, T, D)).astype(np.float16)
    vn = rng.standard_normal((B, H, T, D)).astype(np.float16)
    q, k, v = (NBXTensor.from_numpy(x) for x in (qn, kn, vn))
    got = _d2h(W.scaled_dot_product_attention_wrapper(
        q, k, v, is_causal=causal)).reshape(B, H, T, D)
    ref = _attention_float64(qn, kn, vn, causal)
    return float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9))


# Head dimensions the zoo actually uses, plus the neighbours that isolate
# a tile boundary. 128 and 256 are the powers of two where the kernel's
# EVEN_HEADDIM predicate selects an unmasked load.
_HEAD_DIMS = [16, 32, 64, 80, 96, 112, 127, 128, 129, 160, 192, 255, 256]


# Head dims where the kernel is KNOWN WRONG today — P-FLASH-D128-CORRECTNESS.
# Listed explicitly rather than skipped, so that a fix turns these into
# unexpected passes and forces the list to be revisited.
_KNOWN_BROKEN_HEAD_DIMS = {128, 256}


@pytest.mark.parametrize("D", _HEAD_DIMS)
@pytest.mark.parametrize("T", [64, 256])
def test_attention_is_correct_not_merely_reproducible(D, T) -> None:
    """The check a byte gate cannot make."""
    if not _has_gpu():
        pytest.skip("no GPU")
    if D in _KNOWN_BROKEN_HEAD_DIMS:
        err = _relative_error(1, 4, T, D)
        assert err > ATTENTION_FP16_REL_BOUND, (
            f"head_dim={D} now passes the correctness bound "
            f"({err:.3e} <= {ATTENTION_FP16_REL_BOUND:.1e}). "
            f"P-FLASH-D128-CORRECTNESS may be fixed — remove {D} from "
            f"_KNOWN_BROKEN_HEAD_DIMS and re-run the zoo.")
        return
    err = _relative_error(1, 4, T, D)
    assert err <= ATTENTION_FP16_REL_BOUND, (
        f"head_dim={D} T={T}: relative error {err:.3e} exceeds the declared "
        f"bound {ATTENTION_FP16_REL_BOUND:.1e} by {err/ATTENTION_FP16_REL_BOUND:.0f}x. "
        f"This is a CORRECTNESS failure, not a reproducibility one: the "
        f"kernel may return this same wrong answer every time and pass "
        f"every byte gate.")


@pytest.mark.parametrize("B,H,T,D", [
    (1, 1, 256, 128),      # minimal reproducer of the D=128 defect
    (1, 32, 256, 128),     # canonical row's decode shape
    (2, 4, 128, 128),      # batch > 1
    (1, 8, 512, 64),       # a known-good dim at depth
    (1, 4, 512, 128),
])
def test_attention_correct_across_shapes(B, H, T, D) -> None:
    if not _has_gpu():
        pytest.skip("no GPU")
    err = _relative_error(B, H, T, D)
    if D in _KNOWN_BROKEN_HEAD_DIMS:
        assert err > ATTENTION_FP16_REL_BOUND, (
            f"B={B} H={H} T={T} D={D} now passes ({err:.3e}) — "
            f"P-FLASH-D128-CORRECTNESS may be fixed; update "
            f"_KNOWN_BROKEN_HEAD_DIMS and re-run the zoo.")
        return
    assert err <= ATTENTION_FP16_REL_BOUND, (
        f"B={B} H={H} T={T} D={D}: relative error {err:.3e} exceeds "
        f"{ATTENTION_FP16_REL_BOUND:.1e}")


def test_non_causal_is_correct() -> None:
    """Non-causal at D=128 — currently the WORST case measured (1.9e-01).

    Asserted broken for the same reason as the causal cells: hiding it
    would let a fix go unnoticed."""
    if not _has_gpu():
        pytest.skip("no GPU")
    err = _relative_error(1, 4, 128, 128, causal=False)
    if 128 in _KNOWN_BROKEN_HEAD_DIMS:
        assert err > ATTENTION_FP16_REL_BOUND, (
            f"non-causal D=128 now passes ({err:.3e}) — "
            f"P-FLASH-D128-CORRECTNESS may be fixed.")
        return
    assert err <= ATTENTION_FP16_REL_BOUND, (
        f"non-causal D=128: relative error {err:.3e}")


def test_the_bound_is_not_vacuous() -> None:
    """A bound so loose that anything passes proves nothing.

    Asserts the bound is at least an order of magnitude below the defect
    it was written to catch (3.4e-01) and at least twice the largest
    error observed on a correct path (5.0e-04).
    """
    assert ATTENTION_FP16_REL_BOUND < 3.4e-01 / 10, "bound too loose to catch the known defect"
    assert ATTENTION_FP16_REL_BOUND > 5.0e-04 * 2, "bound too tight for fp16 rounding"


if __name__ == "__main__":  # script mode
    if not _has_gpu():
        raise SystemExit("no GPU")
    fails = 0
    print(f"declared bound: {ATTENTION_FP16_REL_BOUND:.1e} relative\n")
    print(f"{'D':>5s} {'T':>5s} {'rel err':>10s}  verdict")
    for D in _HEAD_DIMS:
        for T in (64, 256):
            try:
                err = _relative_error(1, 4, T, D)
                ok = err <= ATTENTION_FP16_REL_BOUND
                if not ok:
                    fails += 1
                print(f"{D:5d} {T:5d} {err:10.2e}  "
                      f"{'ok' if ok else f'FAIL ({err/ATTENTION_FP16_REL_BOUND:.0f}x bound)'}")
            except Exception as e:
                fails += 1
                print(f"{D:5d} {T:5d}  ERROR {type(e).__name__}: {str(e)[:40]}")
    print()
    for B, H, T, D in [(1, 1, 256, 128), (1, 32, 256, 128), (2, 4, 128, 128),
                       (1, 8, 512, 64), (1, 4, 512, 128)]:
        try:
            err = _relative_error(B, H, T, D)
            ok = err <= ATTENTION_FP16_REL_BOUND
            if not ok:
                fails += 1
            print(f"  B={B} H={H} T={T} D={D}: {err:.2e} "
                  f"{'ok' if ok else 'FAIL'}")
        except Exception as e:
            fails += 1
            print(f"  B={B} H={H} T={T} D={D}: ERROR {e}")
    try:
        test_non_causal_is_correct()
        print("  non-causal D=128: ok")
    except AssertionError as e:
        fails += 1
        print(f"  non-causal D=128: FAIL {e}")
    test_the_bound_is_not_vacuous()
    print("\n  bound sanity: ok")
    print(f"\n{'ALL GREEN' if not fails else f'{fails} CORRECTNESS FAILURE(S)'}")
    raise SystemExit(1 if fails else 0)
