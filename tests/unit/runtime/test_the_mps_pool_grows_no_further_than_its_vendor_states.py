"""The compiled engine bounds a device's torch pool at the fraction its vendor file states.

torch's MPS pool may grow to 1.7 x the device's recommended working set by default and kept the freed buffers of
Allegro's cut attention: 23.7 GB footprint on the M4 Pro's 17.8 GB working set, into swap (the Mac, 2026-10-15 08:30).
Every Apple vendor file states `memory.compiled_allocator_memory_fraction`; the engine applies it through the device's
torch backend at setup.

What this test does if the code is wrong: the vendor cell fails when an Apple file does not state the fraction or a
CUDA one does; the apply cell fails when the backend's call is not made with the stated fraction, when an operator's
PYTORCH_MPS_HIGH_WATERMARK_RATIO does not win, or when two fractions in one plan are not refused.
"""
import glob
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from neurobrix.core.runtime import torch_allocator as TA
from neurobrix.core.config.loader import vendor_file  # the device whole: engine file + library half

VENDORS = Path(__file__).resolve().parents[3] / "src" / "neurobrix" / "config" / "vendors"


def _plan(*allocs):
    return SimpleNamespace(components={f"c{i}": SimpleNamespace(devices=[d], vendor=v, architecture=a)
                                       for i, (d, v, a) in enumerate(allocs)})


def test_apple_states_the_fraction_and_no_other_vendor_does():
    for f in sorted(glob.glob(str(VENDORS / "*" / "*.yml"))):
        m = (vendor_file(f) or {}).get("memory") or {}
        if "/apple/" in f:
            assert 0 < m["compiled_allocator_memory_fraction"] <= 2, f
        else:
            assert "compiled_allocator_memory_fraction" not in m, f


def test_the_fraction_is_applied_through_the_devices_backend(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.mps, "set_per_process_memory_fraction", lambda f: calls.append(f))
    monkeypatch.delenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", raising=False)
    plan = _plan(("mps:0", "apple", "apple_m4_pro"), ("cpu", "apple", "apple_m4_pro"))
    assert TA.configure_memory_fraction(plan) == 1.0 and calls == [1.0]
    assert TA.configure_memory_fraction(_plan(("cuda:0", "nvidia", "volta"))) is None
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.85")
    calls.clear()
    assert TA.configure_memory_fraction(plan) is None and not calls, "the operator's ratio wins"
    with pytest.raises(RuntimeError, match="declare different memory.compiled_allocator_memory_fraction"):
        TA.declared_fraction(_plan(("mps:0", "apple", "apple_m4_pro"), ("cuda:0", "nvidia", "volta")))
