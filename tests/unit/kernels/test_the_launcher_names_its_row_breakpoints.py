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
    """The Volta vendor file in force (`_configs.vendor_profile_for_arch("7.0")`, the compute capability): the ladders and the matrix unit are DATA on it."""
    from neurobrix.kernels import wrappers as W
    from neurobrix.kernels.ops import _configs as CF
    W.set_hardware_profile(profile(V100_16GB))
    monkeypatch.setattr(CF, "_ACTIVE_PROFILE", CF.vendor_profile_for_arch("7.0"))
    yield CF.active_vendor_profile()


def test_the_breakpoints_are_the_gemv_bound_every_bucket_step_and_the_units_take_over(bound_profile):
    from neurobrix.kernels import launch_keys as LK
    from neurobrix.kernels.nbx_tensor import NBXDtype
    bps = LK.row_breakpoints(1, 200)
    assert 5 in bps                                                   # M <= 4: the per-row GEMV, no key
    half, single = NBXDtype.float16, NBXDtype.float32
    def klass(m):      # the launcher's own class of a row count: GEMV bound, M bucket, the unit's take-over per representation
        return (m <= 4, LK.bucket_of("M", m), LK.matrix_unit_takes(half, half, m), LK.matrix_unit_takes(single, half, m))
    units = [m for m in range(1, 201) if LK.matrix_unit_takes(half, half, m)]
    if units:                                                         # a profile with a matrix unit: its take-over is a breakpoint
        assert units[0] in bps
    assert bps == sorted(set(bps)) and all(1 < p <= 200 for p in bps)
    for m in range(2, 201):                                           # every breakpoint changes the class, and nothing else does
        assert (m in bps) == (klass(m) != klass(m - 1)), m
    assert LK.row_breakpoints(10, 10) == [] and LK.row_breakpoints(5, 3) == []
