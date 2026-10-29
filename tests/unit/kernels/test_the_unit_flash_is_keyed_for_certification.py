"""Workstream 1: the flash on the matrix unit (ops/flash_attention_m8n8k4.py) is keyed for certification
(`launch_keys.unit_flash_launches`, derived by the census on the sdpa route "flash"): its tile came from a hand table
of the profile per head dim (volta.yml matrix_unit.flash, the best of six measured on 2026-10-07); keyed, it is
certified per key. What this test would do if the key were wrong: no key where the unit takes the attention, or a key
where it does not -> red."""
import pytest

from tests.unit.prism._pinned_machine import V100_16GB, profile


@pytest.fixture
def volta(monkeypatch):
    from nbxtensor.ops import wrappers as W
    from nbxtensor.ops import _configs as CF
    W.set_hardware_profile(profile(V100_16GB))
    monkeypatch.setattr(CF, "_ACTIVE_PROFILE", CF.vendor_profile_for_arch("7.0"))
    yield


def test_a_half_attention_on_the_unit_forms_its_key(volta):
    from nbxtensor.ops import launch_keys as LK
    from nbxtensor.dtype.nbx_dtype import NBXDtype as D
    (kern, key), = LK.unit_flash_launches(32760, 32760, 128, D.float16, D.float16, D.float16)
    assert kern == LK.UNIT_FLASH
    assert key == (LK.bucket_of("M", 32760), LK.bucket_of("N", 32760), 128, False,
                   "fp16", "fp16", "fp16", "fp16", "fp16", "fp32")


def test_an_attention_the_unit_does_not_take_forms_none(volta):
    from nbxtensor.ops import launch_keys as LK
    from nbxtensor.dtype.nbx_dtype import NBXDtype as D
    assert not LK.unit_flash_takes(512, D.float16, D.float16, D.float16)
    assert LK.unit_flash_launches(4096, 4096, 512, D.float16, D.float16, D.float16) == []


def test_the_candidate_tiles_divide_their_bands_among_their_warps():
    from nbxtensor.ops import flash_attention_m8n8k4 as F
    keep = F._prune_unit_flash(F.unit_flash_configs(), {})
    assert keep and all(c.kwargs["MB"] % c.num_warps == 0 for c in keep)
    assert {(c.kwargs["MB"] * 16, c.kwargs["NS"] * 16, c.num_warps) for c in keep} >= {(128, 64, 4), (128, 32, 4), (64, 32, 4)}
