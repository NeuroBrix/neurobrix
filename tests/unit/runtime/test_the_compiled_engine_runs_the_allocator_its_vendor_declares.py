"""The compiled engine runs torch's caching allocator in the mode its vendor profile declares
(`memory.compiled_allocator_settings`, core/runtime/torch_allocator.py), before a weight loads.

Wan2.1-I2V-14B compiled on a 16 GB V100 died at aten.view_as_complex::4 with the plan's bytes
allocated (8.52 GiB + 2.50 GiB asked, priced 11.0 GiB) and 6.76 GiB reserved-but-unallocated: the
fixed segments' fragmentation, which the allocator's expandable segments remove.

Injections (each seen red, then restored green):
  * the `configure_torch_allocator` call removed from `RuntimeExecutor.setup` -> the compiled
    setup applies nothing;
  * `compiled_allocator_settings` removed from volta.yml -> nothing declared for the V100 plan;
  * removed from amd/cdna3.yml, or added to an Apple profile -> the per-profile test names it;
  * the engine test in `setup` dropped (applied under every mode) -> the Triton setup calls it.
"""
import os
import sys
import types

import pytest

from neurobrix.core.runtime import torch_allocator as TA
from neurobrix.core.runtime.executor import RuntimeExecutor


def _plan(*allocs):
    return types.SimpleNamespace(components={
        f"c{i}": types.SimpleNamespace(devices=d, vendor=v, architecture=a)
        for i, (d, v, a) in enumerate(allocs)})


V100 = (["cuda:0"], "nvidia", "volta")


@pytest.fixture
def setter(monkeypatch):
    import torch
    calls = []
    monkeypatch.setattr(torch._C, "_accelerator_setAllocatorSettings", calls.append, raising=False)  # a stub on any torch
    monkeypatch.setattr(TA, "_applied", None)
    for k in TA.OPERATOR_ENV:
        monkeypatch.delenv(k, raising=False)
    return calls


def _setup(monkeypatch, mode, plan):
    ex = RuntimeExecutor.__new__(RuntimeExecutor)
    ex.mode, ex.plan, ex._is_setup = mode, plan, False
    for name in ("_setup_modules", "_setup_executors", "_init_strategy"):
        monkeypatch.setattr(ex, name, lambda: None)
    ex.setup()


def test_the_compiled_setup_applies_the_vendors_allocator_before_the_executors(monkeypatch, setter):
    order = []
    ex = RuntimeExecutor.__new__(RuntimeExecutor)
    ex.mode, ex.plan, ex._is_setup = "compiled", _plan(V100), False
    for name in ("_setup_modules", "_init_strategy"):
        monkeypatch.setattr(ex, name, lambda: None)
    monkeypatch.setattr(ex, "_setup_executors", lambda: order.append(list(setter)))
    ex.setup()
    assert order == [["expandable_segments:True"]], order     # set before any executor exists


def test_a_torch_without_the_setter_is_refused_naming_the_release_it_needs_and_the_installed_one(monkeypatch, setter):
    # torch 2.5.1 has no torch._C._accelerator_setAllocatorSettings (first released in v2.10.0): an AttributeError was
    # all such a user read; the refusal names both versions, never sets the old environment variable, never skips
    import torch
    monkeypatch.delattr(torch._C, "_accelerator_setAllocatorSettings")
    monkeypatch.setattr(torch, "__version__", "2.5.1")
    with pytest.raises(RuntimeError, match=r"first released in torch 2\.10\.0; the installed torch is 2\.5\.1"):
        TA.configure_torch_allocator(_plan(V100))
    assert TA._applied is None and not any(os.environ.get(k) for k in TA.OPERATOR_ENV)


def test_a_torch_with_the_setter_applies_the_declared_setting(setter):
    assert TA.configure_torch_allocator(_plan(V100)) == "expandable_segments:True"
    assert setter == ["expandable_segments:True"]


def test_the_triton_engine_never_touches_torchs_allocator(monkeypatch, setter):
    def refuse(plan):
        raise AssertionError("the Triton engine reached torch's allocator")
    import neurobrix.core.runtime.executor as E
    monkeypatch.setattr(E, "configure_torch_allocator", refuse)
    for mode in ("triton", "triton_sequential"):
        _setup(monkeypatch, mode, _plan(V100))
    assert setter == []


def _profiles():
    """(vendor, arch) of every vendor profile in the package, read from the directory."""
    from pathlib import Path
    import neurobrix
    root = Path(neurobrix.__file__).resolve().parent / "config" / "vendors"
    return sorted((p.parent.name, p.stem) for p in root.glob("*/*.yml"))


#: The vendors whose torch allocator implements expandable segments: CUDA's, and ROCm's build of
#: the same allocator (hipMemCreate / hipMemMap, ROCm >= 7.0 —
#: https://github.com/pytorch/pytorch/blob/v2.14.0/c10/cuda/CUDAAllocatorConfig.h). Apple's MPS
#: allocator has no such setting (aten/src/ATen/mps/MPSAllocator.mm).
EXPANDABLE = {"nvidia", "amd"}


def test_every_profile_whose_allocator_has_the_setting_declares_one_torch_parses():
    """Injections (seen red, then restored green — STATE.md of the op_tiler campaign, 2026-10-05):
    the declaration removed from amd/cdna3.yml -> red naming amd/cdna3; one added to an Apple
    profile -> red naming it."""
    import torch
    from neurobrix.core.config.loader import get_vendor_config
    profiles = _profiles()
    assert {v for v, _ in profiles} >= EXPANDABLE | {"apple"}, profiles      # not vacuous
    before = torch._C._accelerator_getAllocatorSettings()
    try:
        for vendor, arch in profiles:
            value = (get_vendor_config(vendor, arch).get("memory") or {}).get(TA.PROFILE_KEY)
            if vendor not in EXPANDABLE:
                assert value is None, f"{vendor}/{arch} declares {value!r}; its allocator has none"
                continue
            assert value, f"{vendor}/{arch} declares no memory.{TA.PROFILE_KEY}"
            torch._C._accelerator_setAllocatorSettings(value)        # the real parser
            assert torch._C._accelerator_getAllocatorSettings() == value, (vendor, arch)
    finally:
        torch._C._accelerator_setAllocatorSettings(before or "expandable_segments:False")


@pytest.mark.parametrize("name", TA.OPERATOR_ENV)
def test_every_spelling_torch_reads_is_the_operators(monkeypatch, setter, name):
    """torch reads PYTORCH_CUDA_ALLOC_CONF, PYTORCH_HIP_ALLOC_CONF and PYTORCH_ALLOC_CONF
    (c10/core/AllocatorConfig.cpp); an operator's value under any of them wins on any vendor."""
    assert set(TA.OPERATOR_ENV) == {
        "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF"}
    monkeypatch.setenv(name, "expandable_segments:False")
    assert TA.configure_torch_allocator(_plan((["cuda:0"], "amd", "cdna3"))) is None
    assert setter == []


def test_the_operators_setting_wins_and_a_host_plan_declares_nothing(monkeypatch, setter):
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:False")
    assert TA.configure_torch_allocator(_plan(V100)) is None
    monkeypatch.delenv("PYTORCH_ALLOC_CONF")
    assert TA.configure_torch_allocator(_plan((["cpu"], "nvidia", "volta"))) is None
    assert setter == []
    assert TA.configure_torch_allocator(_plan(V100, (["cpu"], "nvidia", "volta"))) == "expandable_segments:True"
    assert TA.configure_torch_allocator(_plan(V100)) == "expandable_segments:True"
    assert setter == ["expandable_segments:True"]                 # once per process


def test_two_profiles_declaring_two_settings_are_refused_by_name(monkeypatch, setter):
    import neurobrix.core.config.loader as L
    real = L.get_vendor_config

    def fake(vendor, arch):
        if arch == "ampere":
            return {"memory": {TA.PROFILE_KEY: "expandable_segments:False"}}
        return real(vendor, arch)
    monkeypatch.setattr(L, "get_vendor_config", fake)
    with pytest.raises(RuntimeError, match="nvidia/ampere.*|nvidia/volta.*"):
        TA.configure_torch_allocator(_plan(V100, (["cuda:1"], "nvidia", "ampere")))
    assert setter == []
