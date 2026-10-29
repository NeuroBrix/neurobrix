"""A GEMM key is priced and synthesised at a row count whose launch forms it (`launch_keys.gemm_row_representative`).

On Volta the M classes of eight below the matrix unit key the rows 57..64 of an fp32 GEMM as M_BUCKET 64, and at 64
itself the split matrix unit takes the GEMM — no key. The certifier synthesised every key at its bucket's top: 172 keys
of lot (b) FAILED "the wrapper never reached the autotuner" (2026-10-19 13:35). The representative is the largest row
count of the bucket the unit does not take; a bucket the unit takes whole has none and is refused by name.

What this test would do if the code were wrong: the bucket's top returned (64) — red at the launcher check (the launch
at 64 forms no key); a row outside the bucket returned — red at the bucket assertion.
"""
from __future__ import annotations

import pytest

from tests.unit.prism._pinned_machine import V100_16GB, profile


@pytest.fixture
def bound_profile(monkeypatch):
    from nbxtensor.ops import wrappers as W
    from nbxtensor.ops import _configs as CF
    W.set_hardware_profile(profile(V100_16GB))
    monkeypatch.setattr(CF, "_ACTIVE_PROFILE", CF.vendor_profile_for_arch("7.0"))
    yield CF.active_vendor_profile()


def test_every_keyed_bucket_s_representative_forms_its_key(bound_profile):
    from nbxtensor.ops import launch_keys as LK
    from nbxtensor.tensor import NBXDtype as D
    checked = 0
    for a, b in ((D.float32, D.float32), (D.float16, D.float16), (D.float32, D.float16)):
        buckets = sorted({LK.bucket_of("M", m) for m in range(5, 600)})
        for mb in buckets:
            m = LK.gemm_row_representative(mb, a, b)
            if m is None:                                        # the unit takes every row of the bucket: no key
                assert all(LK.matrix_unit_takes(a, b, r) for r in range(5, 600) if LK.bucket_of("M", r) == mb)
                continue
            assert LK.bucket_of("M", m) == mb and not LK.matrix_unit_takes(a, b, m)
            keys = LK.mm_launches(m, 64, 64, a, b, native_bf16=False)
            assert keys and all(k[1][0] == mb for k in keys), (a, b, mb, m, keys)
            checked += 1
    assert checked


def test_the_volta_fp32_class_below_the_unit_is_certified_at_63(bound_profile):
    from neurobrix.kernels import autotune_certify as A
    assert A._gemm_rows((64, 128, 57, True, False, False, "fp32", "fp32", "fp32", "fp32"), ["fp32"] * 4) == 63



@pytest.fixture
def apple_profile(monkeypatch):
    """The Mac's M ladder keys every row count below five on its own (buckets 1, 2, 3, 4)."""
    from tests.unit.prism._pinned_machine import APPLE_M4_PRO
    from nbxtensor.ops import wrappers as W
    from nbxtensor.ops import _configs as CF
    W.set_hardware_profile(profile(APPLE_M4_PRO))
    monkeypatch.setattr(CF, "_ACTIVE_PROFILE", CF.vendor_profile_for_arch("apple-m4-pro"))
    bound = CF.active_vendor_profile()
    assert bound.get("compute_capability") == "apple-m4-pro", "the Apple profile is not the one in force"
    yield bound


def test_a_batched_gemm_of_a_few_rows_is_certified_at_its_own_rows(apple_profile):
    """A decode step's attention is a bmm of one query row per head: `bmm_launches` keys it at M_BUCKET 1 (no GEMV
    route for a batched GEMM). The representative stopped at five rows and refused those keys as census defects (80
    on the Mac, 2026-10-20) — red here with the old bound."""
    from nbxtensor.ops import launch_keys as LK
    from nbxtensor.tensor import NBXDtype as D
    for m in (1, 2, 3, 4):
        for d in (D.float32, D.bfloat16, D.float16):
            keys = LK.bmm_launches(m, 128, 128, d, d, native_bf16=True)
            assert keys, (m, d)
            mb = keys[0][1][0]
            assert mb <= 4, (m, mb)
            r = LK.gemm_row_representative(mb, d, d)
            assert r is not None and LK.bucket_of("M", r) == mb, (m, mb, r)
            assert LK.bmm_launches(r, 128, 128, d, d, native_bf16=True)[0][1] == keys[0][1]
