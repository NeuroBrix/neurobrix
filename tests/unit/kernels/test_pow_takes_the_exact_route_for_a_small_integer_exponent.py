"""`x ** e` for a small integer `e` is within two ulps of the fp64 oracle.

Seen red on 2026-09-20 on CUDA against the kernel that computes every power as
`exp(e * log|x|)`: over 38 147 standard-normal lanes, `x ** 2` reached 15 ulps
from the float64 oracle (main's libdevice call: 2), `x ** -1` 7 ulps (main: 2).
The exact route — repeated multiplication, a reciprocal for a negative exponent —
is what a square IS, and it is what the wrapper now selects at compile time for
an integer-valued exponent with 1 <= |e| <= 8.

The shape is part of the test: the range spans six decades because the exp/log
error grows with |log x| (the Mac measured 4x at |x| in [0.5, 2] and 64x at
[1e-6, 1e-3]); zero lanes and negative lanes are present because IEEE's sign rules
for (-0)**2, (-0)**3 and 1/(+-0) are what the route must keep for free.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("triton")

from neurobrix.kernels.nbx_tensor import NBXTensor  # noqa: E402
import neurobrix.kernels.wrappers as W  # noqa: E402



def _gpu_or_skip():
    try:
        from neurobrix.kernels.nbx_tensor import _detect_gpu_backend
        if _detect_gpu_backend() is None:
            pytest.skip("no GPU backend the engine can resolve")
    except Exception:
        pytest.skip("no GPU backend the engine can resolve")


def _ulps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a.view(np.int32).astype(np.int64) - b.view(np.int32).astype(np.int64))


def _inputs() -> np.ndarray:
    rng = np.random.default_rng(20260920)
    mag = 10.0 ** rng.uniform(-6, 3, size=(37, 1031))          # six decades
    x = (mag * rng.choice([-1.0, 1.0], size=mag.shape)).astype(np.float32)
    x[3, :17] = 0.0                                              # the zero lane
    x[5, 5] = -0.0                                               # and its sign
    return x


@pytest.mark.parametrize("e", [2, 3, 4, -1, -2])
def test_a_small_integer_power_is_within_two_ulps_of_the_oracle(e):
    _gpu_or_skip()
    x = _inputs()
    got = W.pow_wrapper(NBXTensor.from_numpy(x), float(e)).numpy()
    with np.errstate(divide="ignore", over="ignore"):
        want = (x.astype(np.float64) ** e).astype(np.float32)
    fin = np.isfinite(want)
    # non-finite lanes: same infinity, same sign — 1/(+-0) is +-inf
    assert np.array_equal(np.isinf(got), np.isinf(want)), "infinities differ from the oracle"
    assert np.array_equal(np.signbit(got[~fin]), np.signbit(want[~fin])), "the sign of an infinity differs"
    # the sign of a zero result: (-0)**3 is -0, (-0)**2 is +0
    zero = fin & (want == 0.0)
    assert np.array_equal(np.signbit(got[zero]), np.signbit(want[zero])), "the sign of a zero differs"
    u = _ulps(got[fin], want[fin])
    assert u.max() <= 2, f"x**{e}: {u.max()} ulps from the fp64 oracle at worst (mean {u.mean():.3f}); the bound is 2"


def test_a_non_integer_exponent_still_takes_the_general_route():
    """0.5 is not an integer; the general exp/log route serves it. This cell
    guards the DISPATCH — that a non-integer exponent still returns a finite,
    sane value — not the accuracy of that route, which is the Mac's measured
    cost of exp(e log|x|): on this input (six decades, 38 147 lanes) the route
    lands 9 ulps from the oracle at worst, measured 2026-09-20 on CUDA. The bound
    is written from that measurement with room for another backend's exp/log,
    and a route that silently became the exact one would not be caught here —
    the cell above catches the reverse, which is the defect that happened."""
    _gpu_or_skip()
    x = np.abs(_inputs())
    got = W.pow_wrapper(NBXTensor.from_numpy(x), 0.5).numpy()
    want = np.sqrt(x.astype(np.float64)).astype(np.float32)
    assert np.isfinite(got).all()
    assert _ulps(got, want).max() <= 32, "the general route drifted past 32 ulps on x**0.5; measured 9 on CUDA 2026-09-20"
