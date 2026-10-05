"""The compiled engine runs torch's caching allocator in the mode its vendor profile declares
(`memory.compiled_allocator_settings`, core/runtime/torch_allocator.py), before a weight loads.

Wan2.1-I2V-14B compiled on a 16 GB V100 died at aten.view_as_complex::4 with the plan's bytes
allocated (8.52 GiB + 2.50 GiB asked, priced 11.0 GiB) and 6.76 GiB reserved-but-unallocated: the
fixed segments' fragmentation, which the allocator's expandable segments remove.

Injections (each seen red, then restored green):
  * the `configure_torch_allocator` call removed from `RuntimeExecutor.setup` -> the compiled
    setup applies nothing;
  * `compiled_allocator_settings` removed from volta.yml -> nothing declared for the V100 plan;
  * the engine test in `setup` dropped (applied under every mode) -> the Triton setup calls it.
"""
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
    monkeypatch.setattr(torch._C, "_accelerator_setAllocatorSettings", calls.append)
    monkeypatch.setattr(TA, "_applied", None)
    for k in TA.OPERATOR_ENV:
        monkeypatch.delenv(k, raising=False)
    return calls


def _setup(monkeypatch, mode, plan):
    ex = RuntimeExecutor.__new__(RuntimeExecutor)
    ex.mode, ex.plan, ex._is_setup = mode, plan, False
    for name in ("_optimize_cpu_threading", "_setup_modules", "_setup_executors", "_init_strategy"):
        monkeypatch.setattr(ex, name, lambda: None)
    ex.setup()


def test_the_compiled_setup_applies_the_vendors_allocator_before_the_executors(monkeypatch, setter):
    order = []
    ex = RuntimeExecutor.__new__(RuntimeExecutor)
    ex.mode, ex.plan, ex._is_setup = "compiled", _plan(V100), False
    for name in ("_optimize_cpu_threading", "_setup_modules", "_init_strategy"):
        monkeypatch.setattr(ex, name, lambda: None)
    monkeypatch.setattr(ex, "_setup_executors", lambda: order.append(list(setter)))
    ex.setup()
    assert order == [["expandable_segments:True"]], order     # set before any executor exists


def test_the_triton_engine_never_touches_torchs_allocator(monkeypatch, setter):
    def refuse(plan):
        raise AssertionError("the Triton engine reached torch's allocator")
    import neurobrix.core.runtime.executor as E
    monkeypatch.setattr(E, "configure_torch_allocator", refuse)
    for mode in ("triton", "triton_sequential"):
        _setup(monkeypatch, mode, _plan(V100))
    assert setter == []


def test_every_nvidia_profile_declares_a_setting_torch_parses():
    import torch
    from neurobrix.core.config.loader import get_vendor_config
    before = torch._C._accelerator_getAllocatorSettings()
    try:
        for arch in ("volta", "ampere", "hopper"):
            value = get_vendor_config("nvidia", arch)["memory"][TA.PROFILE_KEY]
            torch._C._accelerator_setAllocatorSettings(value)        # the real parser
            assert torch._C._accelerator_getAllocatorSettings() == value
    finally:
        torch._C._accelerator_setAllocatorSettings(before or "expandable_segments:False")


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
