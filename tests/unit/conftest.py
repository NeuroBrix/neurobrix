"""Fixtures shared by the unit tests, and the tools' directory on sys.path.

The tools are scripts: run as `python tools/<tool>.py`, their own directory is first on `sys.path`, and they import
their neighbours by name (`import atelier`, the workshop's roots). A test loads a tool by its file
(`importlib.util.spec_from_file_location`), which does not — so the tools' directory is put where the script's run puts
it. (Since the workshop's roots became `tools/atelier.py`, 2026-10-22, every test that loaded such a tool failed with
`No module named 'atelier'`: about 70 cells of the CPU suite, tools and Forge, red on base.)"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TOOLS = str(Path(__file__).resolve().parents[2] / "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)


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


@pytest.fixture
def needs_device():
    """Skip, said, when this host's profile holds no device — a card-less run (the CPU suite runs with
    CUDA_VISIBLE_DEVICES empty): a cell that launches on a card must skip there, not error in `cuInit` or a malloc.
    Request it FIRST, before any fixture that touches the device."""
    from neurobrix.core.prism.autodetect import load_default_profile
    if not getattr(load_default_profile(), "devices", None):
        pytest.skip("no device in this host's profile (a card-less run)")
