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
    from neurobrix.kernels import wrappers as W
    from neurobrix.kernels.ops import _configs as CF
    W.set_hardware_profile(profile(V100_16GB))
    monkeypatch.setattr(CF, "_ACTIVE_PROFILE", CF.vendor_profile_for_arch("7.0"))
    yield CF.active_vendor_profile()


def test_every_keyed_bucket_s_representative_forms_its_key(bound_profile):
    from neurobrix.kernels import launch_keys as LK
    from neurobrix.kernels.nbx_tensor import NBXDtype as D
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
