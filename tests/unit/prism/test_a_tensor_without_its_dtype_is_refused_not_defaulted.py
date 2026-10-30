"""Prism prices every tensor at its TRACED dtype; a meta without one is refused by name, never priced at a default.

`_compute_size` took a `default_dtype_bytes` and priced a dtype-less tensor at it — 2 bytes by `component_work`'s own
default, the compute width elsewhere: an fp32 tensor that lost its dtype was planned at half its size. No container
reaches it (805 445 tensors of the 49 containers, every one with its dtype, 2026-10-30), so the refusal costs no plan.

What would this file do if the default came back? `test_a_meta_without_dtype_is_refused` would price it and fail."""
from __future__ import annotations

import pytest

from neurobrix.core.prism.profiler import ActivationProfiler


def _sizer():
    return ActivationProfiler.__new__(ActivationProfiler)


def test_a_meta_without_dtype_is_refused():
    with pytest.raises(ValueError, match="carries no traced dtype"):
        _sizer()._compute_size([4, 8], {"shape": [4, 8]})
    with pytest.raises(ValueError, match="carries no traced dtype"):
        _sizer()._compute_size([1], {})


@pytest.mark.parametrize("dtype,width", [("float32", 4), ("bfloat16", 2), ("float16", 2), ("int64", 8)])
def test_a_meta_with_its_dtype_is_priced_at_it(dtype, width):
    assert _sizer()._compute_size([4, 8], {"dtype": dtype}) == 32 * width
