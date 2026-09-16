"""An oracle that builds its reference from non-finite operands vouches for nothing.

Measured on swin2SR-classical-sr-x2-64 (M4 Pro, triton-ext, 2026-09-17): a
convolution whose activation input carried 63181 NaN and 356 Inf — absmax
3.39e+38, FLT_MAX — made the fp64 screen report

    the fp64 oracle contradicts EVERY candidate (18 of 18)

NaN compares unequal to everything, itself included, so every candidate
"disagrees" with such a reference. Two outcomes, both wrong: the screen refuses
the entire space and the run dies, or a candidate slips through and the cache
records `screened: true, screened_by: fp64 oracle` — a stamp earned against
garbage. The caches this campaign collected carry exactly that stamp.

So the screen must decline to screen, not screen badly. Declining routes the key
through `_seat_unscreened`, which records it `screened: false` with the reason
and keeps it out of the certified directory. The engine still runs.

No GPU: the provider is called with plain arrays through a stub tuner.
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels import screen_oracle as SO


class _T:
    """The little of NBXTensor the oracle touches."""
    def __init__(self, arr):
        self._a = np.ascontiguousarray(arr)
        self._device = "cpu"
        self._dtype = None
        self.shape = arr.shape
    def numpy(self):
        return self._a
    def data_ptr(self):
        return id(self)


def _matmul_kernel():          # the oracle table is keyed on __name__
    raise AssertionError("never called")


_matmul_kernel.__name__ = "matmul_kernel"


class _Tuner:
    def __init__(self, nargs):
        self.nargs = nargs
        self.base_fn = _matmul_kernel


def _named(a, b, c):
    return {"a_ptr": _T(a), "b_ptr": _T(b), "c_ptr": _T(c)}


def _buffers(named):
    """(addr, nbytes, dtype) for every live tensor, as the screen snapshots them."""
    return [(t.data_ptr(), t.numpy().nbytes, "fp32") for t in named.values()]


def test_a_finite_reference_is_screened():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = np.arange(8, dtype=np.float32).reshape(4, 2)
    named = _named(a, b, np.zeros((3, 2), np.float32))
    out = SO.provider(_Tuner(named), ("k",), _buffers(named), meta={})
    assert out is not None, "a finite reference must be usable for screening"


@pytest.mark.parametrize("poison", [np.nan, np.inf, -np.inf])
def test_a_non_finite_reference_is_refused_with_its_reason(poison):
    a = np.arange(12, dtype=np.float32).reshape(3, 4).copy()
    a[1, 2] = poison
    b = np.arange(8, dtype=np.float32).reshape(4, 2)
    named = _named(a, b, np.zeros((3, 2), np.float32))

    SO.set_last_refusal(None)
    out = SO.provider(_Tuner(named), ("k",), _buffers(named), meta={})

    assert out is None, (
        "the screen must DECLINE a reference it cannot vouch for, so the key is "
        "recorded unscreened rather than stamped `screened: true`")
    why = SO.last_refusal()
    assert why and "not finite" in why, why
    assert "cannot adjudicate" in why


def test_the_launcher_records_that_reason_rather_than_a_generic_one():
    """The record has to say WHY, or `screened: false` reads as "no oracle"."""
    from neurobrix.kernels import launcher as L
    SO.set_last_refusal("the fp64 reference is not finite (7 of 12 elements "
                        "NaN or Inf), so it cannot adjudicate any candidate")
    reason = L._no_oracle_reason(object())          # an oracle WAS present
    assert "not finite" in reason, reason
