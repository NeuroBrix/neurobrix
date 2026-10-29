"""Fixtures shared by the unit tests."""
from __future__ import annotations

import pytest


@pytest.fixture
def declared_profile(monkeypatch):
    """This machine's hardware profile declared to the Triton surface as a run declares it
    (`triton.dtype.declare_hardware_profile`: the wrappers' profile, the kernels' float64 capability, each device's
    memory), restored after the test. A test that runs a Triton sequence needs it: the hot loop sizes its dead-tensor
    queue from the declared profile's vendor file and refuses by name to run without one. Deliberately not autouse —
    a test that runs with no profile must say why."""
    from nbxtensor.ops import wrappers as W
    from nbxtensor.backend import capabilities as K
    from nbxtensor.backend import device_facts as F
    from neurobrix.core.prism.autodetect import load_default_profile
    from neurobrix.triton.dtype import declare_hardware_profile
    monkeypatch.setattr(W, "_NBX_HW_PROFILE", W._NBX_HW_PROFILE)
    monkeypatch.setattr(W, "_NBX_HAS_NATIVE_BF16", W._NBX_HAS_NATIVE_BF16)
    monkeypatch.setattr(K, "_BACKEND_HAS_FP64", K._BACKEND_HAS_FP64)
    monkeypatch.setattr(F, "_DECLARED_MEMORY", dict(F._DECLARED_MEMORY))
    profile = load_default_profile()
    declare_hardware_profile(profile)
    return profile
