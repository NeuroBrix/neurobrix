"""The Volta profile keys a GEMM's row count M in classes of eight below the matrix unit's 64-row tile (decision
2026-10-19 02:32): a 31-token prompt is keyed 32, a 5-token one 8, 64 stays 64, and above 64 the rows are `default`'s
(65 -> 80). Every other dimension keeps the measured ladder (exact below 64). What this test would do if the code
were wrong: an M ladder missing from the profile, or one that still keys rows exactly, fails the first assertions; an
M ladder diverging from default's above 64 fails the last."""
from __future__ import annotations

import pytest

from tests.unit.prism._pinned_machine import V100_16GB, profile


@pytest.fixture
def bound_profile(monkeypatch):
    """The Volta vendor file in force (`_configs.vendor_profile_for_arch("7.0")`, the compute capability): the ladders are DATA on it."""
    from nbxtensor.ops import wrappers as W
    from nbxtensor.ops import _configs as CF
    W.set_hardware_profile(profile(V100_16GB))
    monkeypatch.setattr(CF, "_ACTIVE_PROFILE", CF.vendor_profile_for_arch("7.0"))
    yield CF.active_vendor_profile()


def test_m_is_keyed_in_classes_of_eight_below_the_unit_and_as_default_above(bound_profile):
    from nbxtensor.ops.autotune_bucket import bucket_of
    assert [bucket_of("M", m) for m in (5, 8, 9, 31, 32, 57, 64)] == [8, 8, 16, 32, 32, 64, 64]
    assert bucket_of("M", 65) == 80 and bucket_of("M", 257) == 288
    assert bucket_of("N", 31) == 31 and bucket_of("K", 57) == 57        # the other dimensions: the measured ladder
    for m in range(65, 3000, 7):
        assert bucket_of("M", m) == bucket_of("N", m), m               # above the unit the rows are default's
