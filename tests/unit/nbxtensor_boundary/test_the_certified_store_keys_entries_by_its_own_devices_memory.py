"""The certified store serves an entry only to a card of the memory it was proven on, and that memory is the library's
own fact: the device's, as an application declares it (`device_facts.declare_device_memory`: a census shadow sees no
card; NeuroBrix declares its profile's devices in `triton.dtype.declare_hardware_profile`) or the driver reports it.
It read a NeuroBrix profile and parsed NeuroBrix profile NAMES through two host services (`memory_class_gb`,
`memory_class_from_profile_id`); no committed proof needed the name (76 133 of 76 133 carry their device record).

What this test would do if the code were wrong: a 32 GB card served a 16 GB proof (red); the size not read from what
was declared (red); a proof with no device record given a size from its profile name (red); the engine's profile not
declared to the library (red)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture
def clean_memory(monkeypatch):
    from nbxtensor.backend import device_facts as F
    monkeypatch.setattr(F, "_DECLARED_MEMORY", {})
    monkeypatch.setattr(F, "_MEASURED_MEMORY", {})
    return F


def _proof(mb=None, name=None):
    machine = {}
    if mb is not None:
        machine["device"] = {"memory_mb": mb}
    if name is not None:
        machine["hardware_profile"] = name
    return {"machine": machine}


def test_the_executing_size_is_the_declared_devices(clean_memory):
    from nbxtensor.ops import autotune_certified as C
    F = clean_memory
    F.declare_device_memory(0, 16384 << 20)
    F.declare_device_memory(1, 32768 << 20)
    assert C.executing_memory_gb(device_idx=0) == 16 and C.executing_memory_gb(device_idx=1) == 32
    assert C.executing_memory_gb() is None, "a rig of mixed sizes has no single size"
    entry = {"proof": _proof(16384), "config": {"a": 1}}
    assert C.entry_for_memory_gb(entry, C.executing_memory_gb(device_idx=0)) is entry
    assert C.entry_for_memory_gb(entry, C.executing_memory_gb(device_idx=1)) is None, "a 16 GB proof served 32 GB"


def test_a_proof_without_its_device_record_serves_no_card():
    from nbxtensor.ops import autotune_certified as C
    assert C.proof_memory_gb(_proof(mb=32510)) == 32
    assert C.proof_memory_gb(_proof(name="auto-v100-16gb-16g")) is None, "a size reconstructed from a profile name"


def test_the_engine_declares_its_profiles_devices(clean_memory, monkeypatch):
    from nbxtensor.ops import wrappers as W
    from neurobrix.triton.dtype import declare_hardware_profile
    monkeypatch.setattr(W, "_NBX_HW_PROFILE", None)
    dev = SimpleNamespace(index=0, memory_mb=16384, brand="nvidia", architecture="volta")
    declare_hardware_profile(SimpleNamespace(devices=[dev], has_native_bf16=False))
    assert clean_memory.declared_device_memory() == {0: 16384 << 20}
