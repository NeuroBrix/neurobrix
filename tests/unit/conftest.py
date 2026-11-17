"""Fixtures shared by the unit tests, and the tools' directory on sys.path.

The tools are scripts: run as `python tools/<tool>.py`, their own directory is first on `sys.path`, and they import
their neighbours by name (`import atelier`, the workshop's roots). A test loads a tool by its file
(`importlib.util.spec_from_file_location`), which does not — so the tools' directory is put where the script's run puts
it. (Since the workshop's roots became `tools/atelier.py`, 2026-10-22, every test that loaded such a tool failed with
`No module named 'atelier'`: about 70 cells of the CPU suite, tools and the container builder, red on base.)"""
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


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_gpu_runtime: skipped, said, where no GPU runtime loads on the machine")


def pytest_runtest_setup(item):
    """A test marked `needs_gpu_runtime` skips, said, on a machine where no GPU runtime loads (CUDA, ROCm/HIP, Level
    Zero, Metal): a test that declares a hardware profile to the library or asks its backend runs the library's runtime
    probe, which refuses by name there (`NoGpuRuntime`). A card-less VM is not a failure of the product; the Dell or
    the Mac runs these. A hook, not a fixture: it runs before any fixture that already touches the runtime."""
    if item.get_closest_marker("needs_gpu_runtime") is None:
        return
    from nbxtensor.backend import runtime
    try:
        runtime.gpu_backend()
    except runtime.NoGpuRuntime as why:
        pytest.skip(f"needs a GPU runtime on this machine: {why}")


@pytest.fixture
def needs_device():
    """Skip, said, when this host's profile holds no device — a card-less run (the CPU suite runs with
    CUDA_VISIBLE_DEVICES empty): a cell that launches on a card must skip there, not error in `cuInit` or a malloc.
    Request it FIRST, before any fixture that touches the device. The library's device door is asked before the
    profile (`tests/_no_device.py`, registered by `tests/conftest.py`): a host whose runtime has no driver
    skips by that name; a driver that is there and refuses is not a skip."""
    reason = sys.modules["_no_device"].absence()
    if reason is not None:
        pytest.skip(reason)
    from neurobrix.core.prism.autodetect import load_default_profile
    if not getattr(load_default_profile(), "devices", None):
        pytest.skip("no device in this host's profile (a card-less run)")


@pytest.fixture
def workshop_configuration():
    """Skip, said, on a machine without the workshop configuration (`tools/atelier.py`): for a test that runs a tool
    as a child process, whose import reads that configuration. The reason names the file and how to provide it."""
    from tests.unit.workshop_configuration import missing
    why = missing()
    if why:
        pytest.skip(why)


@pytest.fixture
def workshop_roles(monkeypatch, tmp_path):
    """`workshop_roles("campaigns", ...)`: those workshop roles pinned to the test's own directory (their
    `NBX_ATELIER_<ROLE>` overrides), for a test that reaches a tool only for what the role is a default of. Every tool
    imported meanwhile is forgotten at the end of the test."""
    from tests.unit.workshop_configuration import forget_the_tools_imported, pinned_roles
    before = set(sys.modules)
    yield pinned_roles(monkeypatch, tmp_path)
    forget_the_tools_imported(before)
