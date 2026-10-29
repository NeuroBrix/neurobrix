"""x[mask] = vals: the number of rows written is the mask's True count, and the op's own contract states it — vals has
one row per True element. A census shadow launches nothing and reads no value, so it takes that count from vals; the
read it replaced returned a count the shadow never computed, and Qwen3-VL's DeepStack write (vals [196, 2048]) formed
one row: every Qwen3-VL coverage cell stood rc 1, its keys never checked (filtered out of every proof's summary). A real
run reads the count and refuses by name a mask that disagrees with vals.

What this test would do if the code were wrong: a shadow that reads the count (wrong extent, red); a real run that
writes past or short of vals without a named refusal (red); a correct scatter changed (red)."""
from __future__ import annotations

import numpy as np
import pytest

from nbxtensor.tensor import NBXTensor
from nbxtensor.ops import index_wrappers as IW


def _t(a):
    return NBXTensor.from_numpy(np.ascontiguousarray(a))


def test_the_scatter_writes_one_row_per_true_element():
    x = np.zeros((1, 6, 4), dtype=np.float32)
    mask = np.array([[0, 1, 0, 1, 1, 0]], dtype=np.uint8)
    vals = np.arange(12, dtype=np.float32).reshape(3, 4) + 1
    out = IW.index_put_wrapper(_t(x), [_t(mask)], _t(vals)).numpy()
    want = x.copy(); want[mask.astype(bool)] = vals
    assert np.array_equal(out, want)


def test_a_mask_that_disagrees_with_its_values_is_refused_by_name():
    x = np.zeros((1, 6, 4), dtype=np.float32)
    mask = np.array([[0, 1, 0, 1, 1, 0]], dtype=np.uint8)
    vals = np.ones((2, 4), dtype=np.float32)
    with pytest.raises(RuntimeError, match="holds 3 True elements and the caller's contract states 2"):
        IW.index_put_wrapper(_t(x), [_t(mask)], _t(vals))


def test_a_census_shadow_takes_the_count_from_the_values(monkeypatch):
    from nbxtensor.ops import launch_observer as LO

    class _Shadow:
        @staticmethod
        def active():
            return True

    monkeypatch.setattr(LO, "_OBSERVER", _Shadow)
    mask = _t(np.zeros(10, dtype=np.uint8))          # a shadow's mask holds no values: all zero here
    rows = IW.nonzero_1d_wrapper(mask, expected=7)
    assert tuple(rows.shape) == (7,), rows.shape
