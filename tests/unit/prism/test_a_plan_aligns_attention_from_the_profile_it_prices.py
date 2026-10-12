"""A plan aligns attention operands from the profile it PRICES, never from the card it runs beside.

Prism's width estimate asks `launch_keys.sdpa_operand_dtypes` how an attention whose q, k, v
disagree in dtype is aligned (`precision.attention_operands`, 481c0c1d). Until 2026-10-12 that
function read only `active_vendor_profile()` — the card THIS process drives. A plan for a V100 made
on a host with CUDA_VISIBLE_DEVICES='' (every CPU-only unit run), or a plan for the Mac made on the
rack, read either nothing ("hardware profile None: precision.attention_operands is None") or the
wrong machine: 30 Prism gates refused their plan with PlanNotComputable at activation profiling.

Now the solver reads the alignment off the profile it was handed
(`triton.dtype.profile_attention_operand_alignment`, every device's vendor profile, a mixed
answer refused) and passes it down the width pass; the launcher alone keeps the active profile.
"""
from __future__ import annotations

import copy

import pytest

from neurobrix.kernels.ops import _configs
from neurobrix.triton.dtype import profile_attention_operand_alignment
from tests.unit.prism._pinned_machine import APPLE_M4_PRO, V100_16GB, profile


@pytest.fixture(autouse=True)
def no_card(monkeypatch):
    """No card in force, whatever this host carries: the answer must come from the profile."""
    monkeypatch.setattr(_configs, "active_vendor_profile", lambda: {})


@pytest.mark.parametrize("spec", [V100_16GB, APPLE_M4_PRO], ids=["v100-16gb", "apple-m4-pro"])
def test_the_alignment_is_the_priced_profiles_with_no_card_in_force(spec):
    """Injection: the reader returning `attention_operand_alignment(active_vendor_profile())`
    -> ValueError "hardware profile None", RED."""
    assert profile_attention_operand_alignment(profile(spec)) == "narrowest"


def test_a_profile_with_no_device_has_no_alignment():
    assert profile_attention_operand_alignment(None) is None
    spec = copy.deepcopy(V100_16GB)
    spec["id"] = "scenario-host-only"
    spec["devices"] = []
    assert profile_attention_operand_alignment(profile(spec)) is None


def test_devices_that_disagree_are_refused(monkeypatch):
    """A profile whose devices declare two alignments prices neither (one plan, one answer)."""
    from neurobrix.core.config import loader
    real = loader.get_vendor_config

    def patched(vendor, arch):
        cfg = copy.deepcopy(real(vendor, arch))
        if arch == "apple_silicon":
            cfg.setdefault("precision", {})["attention_operands"] = "widest"
        return cfg

    monkeypatch.setattr(loader, "get_vendor_config", patched)
    spec = copy.deepcopy(V100_16GB)
    spec["id"] = "scenario-mixed"
    spec["devices"] = [V100_16GB["devices"][0], dict(APPLE_M4_PRO["devices"][0], index=1)]
    with pytest.raises(ValueError, match="disagree on precision.attention_operands"):
        profile_attention_operand_alignment(profile(spec))
