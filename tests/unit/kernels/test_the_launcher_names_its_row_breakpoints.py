"""`launch_keys.row_breakpoints`: the row counts at which a GEMM's launch class changes on the bound profile — the
GEMV bound (M <= 4, no key), every M bucket (exact below the matrix unit's smallest tile, then the ladder's steps) and
the matrix unit's take-over (no key from there on). The derived census bands a prompt-length extent by them: a
bisection over the whole extent took the empty key set at both ends (M <= 4 and M >= 64) for ONE class and missed every
keyed prompt length between (TinyLlama, 2026-10-19: a 31-token prompt's matmul key absent under certified-only).
What this test would do if the code were wrong: breakpoints missing 5 or 64, or listing a row inside a constant band,
fail the set assertions; a census banding on nothing is the empty list for an empty extent."""
from __future__ import annotations

import pytest

from tests.unit.prism._pinned_machine import V100_16GB, profile


@pytest.fixture
def bound_profile(monkeypatch):
    from neurobrix.kernels import census as CS, wrappers as W
    prof = profile(V100_16GB)
    W.set_hardware_profile(prof)
    yield prof


def test_the_breakpoints_are_the_gemv_bound_every_bucket_step_and_the_units_take_over(bound_profile):
    from neurobrix.kernels import launch_keys as LK
    from neurobrix.kernels.nbx_tensor import NBXDtype
    bps = LK.row_breakpoints(1, 200)
    assert 5 in bps                                                   # M <= 4: the per-row GEMV, no key
    unit = next(m for m in range(1, 201) if LK.matrix_unit_takes(NBXDtype.float16, NBXDtype.float16, m))
    assert unit in bps                                                # the matrix unit takes over: no key from there on
    assert bps == sorted(set(bps)) and all(1 < p <= 200 for p in bps)
    for p in bps:                                                     # every breakpoint changes the class, and nothing else does
        assert LK.bucket_of("M", p) != LK.bucket_of("M", p - 1) or p == 5 or p == unit \
            or LK.matrix_unit_takes(NBXDtype.float32, NBXDtype.float16, p) != LK.matrix_unit_takes(NBXDtype.float32, NBXDtype.float16, p - 1)
    for m in range(6, 200):
        if m not in bps:
            assert LK.bucket_of("M", m) == LK.bucket_of("M", m - 1)
    assert LK.row_breakpoints(10, 10) == [] and LK.row_breakpoints(5, 3) == []
