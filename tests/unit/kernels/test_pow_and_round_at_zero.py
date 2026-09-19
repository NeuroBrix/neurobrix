"""The zero lane of `pow` and the sign of zero in `round`, against an fp64 oracle.

Both defects were measured on CUDA by the Dell against an fp64 oracle, in the
commit that removed `libdevice` from these kernels (`85c6426a`). They are MINE:
replacing a vendor intrinsic means reimplementing its whole domain, and I
reimplemented the interesting part and not the edges.

  * `pow`: `tl.where(x > 0, mag, tl.where(x < 0, neg, 0.0))` — the `x == 0` lane
    returns 0.0 for EVERY exponent. C99/IEEE says `pow(±0, 0) = 1`,
    `pow(±0, e<0) = ±inf`, and only `pow(±0, e>0)` is a zero.
  * `round`: `floor(x + 0.5)` maps every x in (-0.5, 0] to `+0.0`, losing the
    sign. `round(-0.5)` must be `-0.0` (ties-to-even AND the sign of the input),
    and so must `round(-0.3)` and `round(-0.0)`.

The oracle is numpy in float64, which is not the engine and does not share its
code — an external instrument, per R29.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("triton")

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype  # noqa: E402
import neurobrix.kernels.wrappers as W  # noqa: E402


def _dev_or_skip():
    try:
        from neurobrix.kernels.nbx_tensor import _detect_gpu_backend
        if _detect_gpu_backend() is None:
            pytest.skip("no GPU backend the engine can resolve")
    except Exception:
        pytest.skip("no GPU backend the engine can resolve")


def _same(got: np.ndarray, want: np.ndarray, where: str) -> None:
    """Equal value AND equal sign of zero AND equal infinity — nan == nan."""
    bad = []
    for i, (g, w) in enumerate(zip(got.ravel(), want.ravel())):
        if np.isnan(w):
            ok = np.isnan(g)
        elif w == 0.0:
            ok = (g == 0.0) and (np.signbit(g) == np.signbit(w))
        else:
            ok = (g == w) or (np.isinf(w) and np.isinf(g) and np.signbit(g) == np.signbit(w))
        if not ok:
            bad.append(f"[{i}] got {g!r} (signbit {np.signbit(g)}) "
                       f"want {w!r} (signbit {np.signbit(w)})")
    assert not bad, f"{where}:\n  " + "\n  ".join(bad)


# ---------------------------------------------------------------- pow, x == 0
# THE ORACLE HERE IS THE SPEC, NOT A LIBRARY, and that is a measured decision.
#
# C99 Annex F / IEEE-754 for pow at zero:
#     pow(+-0, 0)                       1
#     pow(+-0, y>0)                     +0, and -0 only for -0 with y an ODD INT
#     pow(+-0, y<0)                     +inf, and -inf only for -0 with y an ODD INT
#
# numpy and torch both diverge from that for a NEGATIVE zero raised to a
# non-integer exponent, and torch is not self-consistent about it. Measured
# 2026-09-19 on x = -0.0:
#
#     e         0.5    1.0    1.5    2.0    2.5    3.0   -0.5   -1.0   -1.5
#     C99        +0     -0     +0     +0     +0     -0   +inf   -inf   +inf
#     torch      -0     -0     +0     +0     +0     -0   -inf   -inf   +inf
#
# torch gives -0 for e=0.5 and +0 for e=1.5 and e=2.5. No rule produces that;
# it is an artifact of however torch decomposes the power. A contract cannot be
# written against it, so this file asserts the spec, which is self-consistent
# and which numpy and torch BOTH match on every case that matters — 0**0,
# 0**(negative), and every integer exponent, including the two the Dell
# measured. The divergence is confined to a signed zero under a non-integer
# exponent, which no model in the catalogue produces.
def _c99_pow_at_zero(signbit_x: bool, e: float) -> float:
    if e == 0.0:
        return 1.0
    odd_int = float(e).is_integer() and (int(e) % 2 != 0)
    neg = signbit_x and odd_int
    if e > 0.0:
        return -0.0 if neg else 0.0
    return float("-inf") if neg else float("inf")


ZERO_EXPONENTS = [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0, 0.5, 1.5, -0.5]


@pytest.mark.parametrize("e", ZERO_EXPONENTS)
def test_pow_of_zero_matches_the_c99_spec(e):
    _dev_or_skip()
    x = np.array([0.0, -0.0], dtype=np.float32)
    want = np.array([_c99_pow_at_zero(False, e), _c99_pow_at_zero(True, e)],
                    dtype=np.float64)
    got = W.pow_wrapper(NBXTensor.from_numpy(x), e).numpy().astype(np.float64)
    _same(got, want, f"pow(x, {e}) on x = [+0.0, -0.0]")


@pytest.mark.parametrize("e", [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
def test_the_integer_exponents_agree_with_numpy_too(e):
    """Where the spec and the libraries agree, check against the library as well.

    This is the set the Dell measured on CUDA, and it is the set that matters:
    every integer exponent plus 0**0. If this ever diverges from numpy, the
    engine and the world disagree about something real.
    """
    _dev_or_skip()
    x = np.array([0.0, -0.0], dtype=np.float32)
    want = np.power(x.astype(np.float64), np.float64(e))
    got = W.pow_wrapper(NBXTensor.from_numpy(x), e).numpy().astype(np.float64)
    _same(got, want, f"pow(x, {e}) against numpy on x = [+0.0, -0.0]")


def test_pow_keeps_the_nonzero_domain_it_already_had():
    """The fix must not disturb the lanes that were right."""
    _dev_or_skip()
    x = np.array([1.0, 2.0, -2.0, 0.5, -0.5, 4.0, -1.0], dtype=np.float32)
    for e in (0.0, 1.0, 2.0, 3.0, -1.0, 0.5):
        want = np.power(x.astype(np.float64), np.float64(e))
        got = W.pow_wrapper(NBXTensor.from_numpy(x), e).numpy().astype(np.float64)
        # non-integer exponent of a negative base is NaN in both
        for i, (g, w) in enumerate(zip(got, want)):
            if np.isnan(w):
                assert np.isnan(g), f"pow({x[i]}, {e}): got {g}, want nan"
            else:
                assert abs(g - w) <= 3e-6 * max(1.0, abs(w)), \
                    f"pow({x[i]}, {e}): got {g}, want {w}"


# ------------------------------------------------------- round, sign of zero
def test_round_keeps_the_sign_of_a_zero_result():
    _dev_or_skip()
    x = np.array([-0.5, -0.3, -0.0, 0.0, 0.3, 0.5, -0.49999997], dtype=np.float32)
    want = np.round(x.astype(np.float64))          # numpy rounds half to even
    got = W.round_wrapper(NBXTensor.from_numpy(x)).numpy().astype(np.float64)
    _same(got, want, "round on values whose result is a zero")


def test_round_still_gets_every_tie_to_even():
    """The sign fix must not disturb the ties, which were already right."""
    _dev_or_skip()
    x = np.array([0.5, 1.5, 2.5, 3.5, 4.5, -1.5, -2.5, -3.5, -4.5,
                  0.49, 1.49, 2.51, -2.51], dtype=np.float32)
    want = np.round(x.astype(np.float64))
    got = W.round_wrapper(NBXTensor.from_numpy(x)).numpy().astype(np.float64)
    _same(got, want, "round ties-to-even")
