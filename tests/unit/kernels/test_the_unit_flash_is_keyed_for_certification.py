"""Workstream 1: the flash on the matrix unit (ops/flash_attention_m8n8k4.py) is keyed for certification
(`launch_keys.unit_flash_launches`, derived by the census on the sdpa route "flash"): its tile came from a hand table
of the profile per head dim (volta.yml matrix_unit.flash, the best of six measured on 2026-10-07); keyed, it is
certified per key. What this test would do if the key were wrong: no key where the unit takes the attention, or a key
where it does not -> red."""
import pytest

from tests.unit.prism._pinned_machine import V100_16GB, profile


@pytest.fixture
def volta(monkeypatch):
    from neurobrix.kernels import wrappers as W
    from neurobrix.kernels.ops import _configs as CF
    W.set_hardware_profile(profile(V100_16GB))
    monkeypatch.setattr(CF, "_ACTIVE_PROFILE", CF.vendor_profile_for_arch("7.0"))
    yield


def test_a_half_attention_on_the_unit_forms_its_key(volta):
    from neurobrix.kernels import launch_keys as LK
    from neurobrix.kernels.nbx_tensor import NBXDtype as D
    (kern, key), = LK.unit_flash_launches(32760, 32760, 128, D.float16, D.float16, D.float16)
    assert kern == LK.UNIT_FLASH
    assert key == ("native", 128, LK.bucket_of("M", 32760), LK.bucket_of("N", 32760), "fp16", "fp16", "fp16")


def test_an_attention_the_unit_does_not_take_forms_none(volta):
    from neurobrix.kernels import launch_keys as LK
    from neurobrix.kernels.nbx_tensor import NBXDtype as D
    assert not LK.unit_flash_takes(512, D.float16, D.float16, D.float16)
    assert LK.unit_flash_launches(4096, 4096, 512, D.float16, D.float16, D.float16) == []
