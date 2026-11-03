"""A GEMM the matrix unit can take runs on the unit or on its FMA kernel as its CERTIFICATE says, never by a threshold.

The list of what remains (2026-11-03, 1.8): `mm` and `addmm` took the m8n8k4 unit whenever the operands had a
representation and M filled the smallest tile (`launch_keys.matrix_unit_operands`), at the first profile tile that
fit — a launch with no key, so the certification never compared the unit with the FMA kernel. Now the FMA launch
keeps its key on the unit too, the certifier times the unit's tiles beside the FMA configurations against the same
oracle (`autotune_certify._unit_runs`), and the entry names the unit's tile only when it was measured faster
(`autotune_certified.unit_choice`, `linalg_wrappers._gemm_route`). A batched product keeps its unit route (its dtypes
differ between the two routes) and its no-key rule.

What these tests would do if the code were wrong: the unit's early return left in `mm_launches` — the census keys
nothing for M >= 64 (red at the first test); the route read from the threshold — the FMA certificate is ignored (red
at the route tests); a unit choice slower than the FMA kernel, or without its tile, accepted by the directory's gate
(red at the validation tests). The device test certifies one fp16 key and reads the choice back.
"""
from __future__ import annotations

import pytest

from tests.unit.prism._pinned_machine import V100_16GB, profile


@pytest.fixture
def volta(monkeypatch):
    from nbxtensor.ops import wrappers as W
    from nbxtensor.ops import _configs as CF
    W.set_hardware_profile(profile(V100_16GB))
    monkeypatch.setattr(CF, "_ACTIVE_PROFILE", CF.vendor_profile_for_arch("7.0"))
    yield CF.active_vendor_profile()


def test_the_fma_key_exists_where_the_unit_can_take_the_gemm(volta):
    from nbxtensor.ops import launch_keys as LK
    from nbxtensor.dtype.nbx_dtype import NBXDtype as D
    assert LK.matrix_unit_takes(D.float16, D.float16, 128)
    assert LK.mm_launches(128, 256, 256, D.float16, D.float16, native_bf16=False)
    assert LK.addmm_launches(128, 256, 256, D.float16, D.float16, D.float16, native_bf16=False)
    assert LK.mm_launches(256, 512, 512, D.float32, D.float32, native_bf16=False)      # the split representation too
    assert LK.bmm_launches(128, 256, 256, D.float16, D.float16, native_bf16=False) == []


def test_every_row_of_a_unit_keyed_bucket_forms_its_key(volta):
    from nbxtensor.ops import launch_keys as LK
    from nbxtensor.dtype.nbx_dtype import NBXDtype as D
    mb = LK.bucket_of("M", 100)
    m = LK.gemm_row_representative(mb, D.float16, D.float16, unit_keyed=True)
    assert m == mb and LK.mm_launches(m, 64, 64, D.float16, D.float16, native_bf16=False)[0][1][0] == mb
    assert LK.gemm_row_representative(mb, D.float16, D.float16) is None      # a batched GEMM's bucket: the unit's


_UNIT = {"block_m": 128, "block_n": 64, "block_k": 32, "warps_m": 4, "warps_n": 1, "group_m": 8,
         "split_a": False, "split_b": False}


@pytest.fixture
def route(monkeypatch):
    from nbxtensor.ops import linalg_wrappers as LW
    from nbxtensor.ops import autotune_certified as C
    from nbxtensor.ops import launch_observer as LO
    monkeypatch.delenv("NBX_AUTOTUNE_CERTIFIED_ONLY", raising=False)
    C.certifying(False)
    monkeypatch.setattr(LO, "launch_observer", lambda: type("O", (), {"active": staticmethod(lambda: False)})())
    return LW, C, monkeypatch


def test_the_certificate_s_unit_tile_is_served(route):
    LW, C, mp = route
    tile = {f: v for f, v in _UNIT.items() if not f.startswith("split")} | {"block_m": 64, "warps_m": 2, "warps_n": 2}
    mp.setattr(C, "unit_choice", lambda t, a, k: ("unit", dict(tile)))
    got = LW._gemm_route(object(), (), {}, dict(_UNIT))
    assert got == {**tile, "split_a": False, "split_b": False}


def test_the_certificate_s_fma_choice_runs_the_fma_kernel(route):
    LW, C, mp = route
    mp.setattr(C, "unit_choice", lambda t, a, k: ("fma", None))
    assert LW._gemm_route(object(), (), {}, dict(_UNIT)) is None


def test_no_certificate_takes_the_declared_unit_but_a_confirmation_run_the_fma_launch(route):
    LW, C, mp = route
    mp.setattr(C, "unit_choice", lambda t, a, k: None)
    assert LW._gemm_route(object(), (), {}, dict(_UNIT)) == _UNIT
    mp.setenv("NBX_AUTOTUNE_CERTIFIED_ONLY", "1")
    assert LW._gemm_route(object(), (), {}, dict(_UNIT)) is None        # the FMA launch then refuses the missing key


def test_the_certifier_s_sweep_takes_the_fma_kernel(route):
    LW, C, mp = route
    mp.setattr(C, "unit_choice", lambda t, a, k: ("unit", dict(_UNIT)))
    C.certifying(True)
    try:
        assert LW._gemm_route(object(), (), {}, dict(_UNIT)) is None
    finally:
        C.certifying(False)


def _doc(unit):
    entry = {"config": {"kwargs": {"BLOCK_M": 64}, "num_warps": 4, "num_stages": 2},
             "proof": {"date": "2026-11-03T20:00:00+00:00", "deviation": 1e-4, "tolerance": 2e-2,
                       "shape": [128, 256, 256, True, False, "fp16", "fp16", "fp32"], "engine_version": "t",
                       "backend": {"triton": "3.8.0", "name": "cuda"}, "machine": {}, "oracle": "fp64", "built": {}},
             "unit": unit}
    return entry


def test_a_unit_choice_slower_than_the_fma_kernel_is_refused():
    from nbxtensor.ops import autotune_certified as C
    tile = {f: v for f, v in _UNIT.items() if not f.startswith("split")}
    probs = []
    C._validate_certification("e", _doc({"tile": tile, "deviation": 1e-4, "best_ms": 0.5, "fma_best_ms": 0.4}),
                              "nbx-autotune-certified/2", None, [128, 256, 256, True, False, "fp16", "fp16", "fp32"],
                              "fp32", probs)
    assert any("matrix unit chosen" in p for p in probs), probs
    probs = []
    C._validate_certification("e", _doc({"tile": {"block_m": 64}, "deviation": 1e-4, "best_ms": 0.3, "fma_best_ms": 0.4}),
                              "nbx-autotune-certified/2", None, [128, 256, 256, True, False, "fp16", "fp16", "fp32"],
                              "fp32", probs)
    assert any("without its tile" in p for p in probs), probs
    probs = []
    C._validate_certification("e", _doc({"tile": tile, "deviation": 1e-4, "best_ms": 0.3, "fma_best_ms": 0.4}),
                              "nbx-autotune-certified/2", None, [128, 256, 256, True, False, "fp16", "fp16", "fp32"],
                              "fp32", probs)
    assert not [p for p in probs if "unit" in p], probs


def _card():
    from nbxtensor.memory.allocator import DeviceAllocator
    return DeviceAllocator.device_count() > 0


@pytest.mark.skipif(not _card(), reason="needs a device to run the kernels")
def test_a_unit_gemm_key_is_certified_with_both_routes_measured():
    import numpy as np
    from nbxtensor.ops import launch_keys as LK
    from nbxtensor.ops import autotune_certified as C
    from nbxtensor.ops.matmul import matmul_kernel
    from neurobrix.kernels import autotune_certify as A
    from nbxtensor.dtype.nbx_dtype import NBXDtype as D
    A._bind_hardware_profile()                      # the card's own profile, as the CLI binds it
    if not LK.matrix_unit_takes(D.float16, D.float16, 128):
        pytest.skip("this card's profile declares no matrix unit")
    (_q, key), = LK.mm_launches(128, 256, 256, D.float16, D.float16, native_bf16=False)
    qual = "nbxtensor.ops.matmul.matmul_kernel"
    cert = A.certify_key(qual, matmul_kernel, key, 2e-2, np.random.default_rng(0))
    measured = cert["proof"]["unit_measured"]
    assert measured["tiles"] and measured["fma_best_ms"] == cert["proof"]["best_ms"]
    best_unit = min(t["ms"] for t in measured["tiles"])
    if best_unit < measured["fma_best_ms"]:
        assert cert["unit"]["best_ms"] == best_unit and set(C._UNIT_TILE_FIELDS) <= set(cert["unit"]["tile"])
    else:
        assert "unit" not in cert
