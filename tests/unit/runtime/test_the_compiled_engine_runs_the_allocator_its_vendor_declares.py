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

torch 2.10 and 2.11 have the setter and lack `torch.accelerator.empty_host_cache` (first in v2.12.0), which the loader
calls after every cuda/hip load: one refusal (`require_compiled_torch`) names both APIs and the `pytorch` extra's
minimum, read from the package metadata, before the setter and before any weight loads.
"""
import os
import importlib.metadata
import sys
import types
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:            # python 3.10
    import tomli as tomllib

from neurobrix.core.runtime import torch_allocator as TA
from neurobrix.core.runtime.executor import RuntimeExecutor
from neurobrix.core.io.weight_loader import WeightLoader

ROOT = Path(__file__).resolve().parents[3]


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
    if hasattr(torch, "accelerator"):
        monkeypatch.setattr(torch.accelerator, "empty_host_cache", lambda: calls.append("empty_host_cache"), raising=False)
    else:                                                          # torch < 2.6 has no torch.accelerator at all
        monkeypatch.setattr(torch, "accelerator", types.SimpleNamespace(empty_host_cache=lambda: calls.append(
            "empty_host_cache")), raising=False)
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


@pytest.fixture
def metadata(monkeypatch):
    """The installed package's requirements, as pip writes them from pyproject.toml (this tree is not installed)."""
    from packaging.requirements import Requirement
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    lines = []
    for e, reqs in extras.items():
        for r in map(Requirement, reqs):
            on = f'({r.marker}) and extra == "{e}"' if r.marker else f'extra == "{e}"'
            lines.append(f"{r.name}{r.specifier}; {on}")
    monkeypatch.setattr(importlib.metadata, "requires", lambda name: lines if name == "neurobrix" else None)
    return lines


#: (sys.platform, platform.system(), platform.machine()) of the four platforms PyPI's torch ships for, and the torch
#: the pytorch extra asks there: 2.12.0 where the compiled engine reaches the GPU calls, 0.6.16's 2.1.0 on macOS (no
#: torch 2.12 wheel for an Intel Mac or a macOS older than 14, and no GPU there that reaches those calls).
PLATFORMS = [("darwin", "Darwin", "x86_64", ">=2.1.0"), ("darwin", "Darwin", "arm64", ">=2.1.0"),
             ("linux", "Linux", "x86_64", ">=2.12.0"), ("win32", "Windows", "AMD64", ">=2.12.0")]


def _on(monkeypatch, sys_platform, system, machine):
    """This process seen as another platform: the marker environment packaging evaluates (it caches the real one for
    the process) and the sys.platform the refusal names."""
    import packaging.markers
    env = dict(packaging.markers.default_environment(), sys_platform=sys_platform, platform_system=system,
               platform_machine=machine)
    monkeypatch.setattr(packaging.markers, "default_environment", lambda: dict(env))
    monkeypatch.setattr(sys, "platform", sys_platform)


@pytest.mark.parametrize("sys_platform,system,machine,need", PLATFORMS)
def test_the_metadata_read_picks_the_torch_requirement_whose_platform_marker_holds(monkeypatch, metadata, sys_platform,
                                                                                    system, machine, need):
    _on(monkeypatch, sys_platform, system, machine)
    assert str(TA.pytorch_extra_torch().specifier) == need


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("machine", ["x86_64", "arm64"])
@pytest.mark.parametrize("via,ext", [("load_component", ".safetensors"), ("load_component_with_shard_map", ".bin")])
def test_on_macos_a_torch_at_2_2_2_loads_on_the_cpu_and_the_apple_gpu(monkeypatch, tmp_path, setter, metadata, machine,
                                                                     device, via, ext):
    # torch 2.2.2 is the last torch PyPI ships for an Intel Mac: neither GPU call is there, and none is reached
    _on(monkeypatch, "darwin", "Darwin", machine)
    loaded = []
    ld = _loader(monkeypatch, tmp_path, loaded, ext)
    _torch_at(monkeypatch, "2.2.2", setter=False, host_cache=False)
    assert _load(ld, via, device) == {"w": device} and setter == []
    assert TA.configure_torch_allocator(_plan(([device], "apple", "apple_m1"))) is None


def test_on_macos_a_torch_meeting_the_extra_that_reaches_the_gpu_calls_is_refused_never_told_to_upgrade(monkeypatch,
                                                                                                     setter, metadata):
    # only another machine's --hardware profile puts a cuda/hip device in a Mac's plan: the extra's torch is met there
    _on(monkeypatch, "darwin", "Darwin", "x86_64")
    _torch_at(monkeypatch, "2.2.2", setter=False, host_cache=False)
    with pytest.raises(RuntimeError, match=r"2\.2\.2 lacks; neurobrix's pytorch extra asks for torch>=2\.1\.0 on "
                       r"darwin, which it meets: .*another machine's --hardware") as exc:
        TA.configure_torch_allocator(_plan(V100))
    assert "upgrade" not in str(exc.value) and setter == []


def _torch_at(monkeypatch, version, setter=True, host_cache=True):
    import torch
    if not setter:
        monkeypatch.delattr(torch._C, "_accelerator_setAllocatorSettings")
    if not host_cache:
        monkeypatch.delattr(torch.accelerator, "empty_host_cache")
    monkeypatch.setattr(torch, "__version__", version)


def test_a_torch_without_the_setter_is_refused_naming_the_release_it_needs_and_the_installed_one(monkeypatch, setter,
                                                                                                metadata):
    # torch 2.5.1 has neither API: an AttributeError was all such a user read; the refusal names both APIs, both
    # versions and the upgrade line, never sets the old environment variable, never skips
    _torch_at(monkeypatch, "2.5.1", setter=False, host_cache=False)
    with pytest.raises(RuntimeError, match=r"calls torch\._C\._accelerator_setAllocatorSettings, torch\.accelerator\."
                       r"empty_host_cache, which the installed torch 2\.5\.1 lacks; neurobrix's pytorch extra asks for "
                       r"torch>=2\.12\.0: upgrade it, pip install --upgrade \"torch>=2\.12\.0\""):
        TA.configure_torch_allocator(_plan(V100))
    assert TA._applied is None and not any(os.environ.get(k) for k in TA.OPERATOR_ENV) and setter == []


def test_a_torch_with_the_setter_and_no_host_cache_release_is_refused_before_the_setting(monkeypatch, setter, metadata):
    # torch 2.11 passed 0.6.17's first refusal (the setter is there) and died in an AttributeError at the first GPU load
    _torch_at(monkeypatch, "2.11.0", host_cache=False)
    with pytest.raises(RuntimeError, match=r"calls torch\.accelerator\.empty_host_cache, which the installed torch "
                       r"2\.11\.0 lacks; neurobrix's pytorch extra asks for torch>=2\.12\.0"):
        TA.configure_torch_allocator(_plan(V100))
    assert setter == [] and TA._applied is None


def _loader(monkeypatch, tmp_path, loaded, ext):
    """A loader whose container is a cache holding one weight file: the real per-file path, the transfer stubbed."""
    ld = WeightLoader.__new__(WeightLoader)
    ld._zip, ld.use_cache, ld._cache_path, ld.file = object(), True, tmp_path, f"components/c/weights/a{ext}"
    (tmp_path / ld.file).parent.mkdir(parents=True)
    (tmp_path / ld.file).write_bytes(b"")
    monkeypatch.setattr(ld, "_verify_weights_present", lambda *a: None)
    monkeypatch.setattr(ld, "_find_weight_files", lambda c: [ld.file])
    monkeypatch.setattr(ld, "_resolve_shard_path", lambda p, c, f: p)
    monkeypatch.setattr(ld, "_get_file_list", lambda: [])
    monkeypatch.setattr(ld, "_refuse_undelivered", lambda *a: None)
    for name in ("_load_with_pinned_dma", "_load_pytorch_with_pinned_dma"):
        monkeypatch.setattr(ld, name, lambda f, d, t, is_cuda: loaded.append((f, d)) or {"w": d})
    import torch
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d=None: None)
    monkeypatch.setattr("neurobrix.core.io.weight_loader.device_sync", lambda d: None)
    return ld


def _load(ld, via, device):
    if via == "load_component":
        return ld.load_component("c", device)
    return ld.load_component_with_shard_map("c", {ld.file: device})


LOADS = [(via, ext) for via in ("load_component", "load_component_with_shard_map") for ext in (".safetensors", ".bin")]


@pytest.mark.parametrize("via,ext", LOADS)
@pytest.mark.parametrize("sys_platform,system,machine", [("linux", "Linux", "x86_64"), ("win32", "Windows", "AMD64")])
def test_a_torch_without_the_host_cache_release_is_refused_before_any_weight_loads_on_a_gpu(monkeypatch, tmp_path,
                                                                                          setter, metadata, via, ext,
                                                                                          sys_platform, system, machine):
    _on(monkeypatch, sys_platform, system, machine)
    loaded = []
    ld = _loader(monkeypatch, tmp_path, loaded, ext)
    _torch_at(monkeypatch, "2.11.0", host_cache=False)
    with pytest.raises(RuntimeError, match=r"empty_host_cache, which the installed torch 2\.11\.0 lacks; .*torch>=2\.12\.0"):
        _load(ld, via, "cuda:0")
    assert loaded == [] and setter == []


@pytest.mark.parametrize("via,ext", LOADS)
def test_a_torch_with_both_apis_loads_onto_a_gpu_and_releases_its_host_cache(monkeypatch, tmp_path, setter, via, ext):
    loaded = []
    ld = _loader(monkeypatch, tmp_path, loaded, ext)
    _torch_at(monkeypatch, "2.12.0")
    assert _load(ld, via, "cuda:0") == {"w": "cuda:0"}
    assert loaded == [(str(tmp_path / ld.file), "cuda:0")] and setter == ["empty_host_cache"]


@pytest.mark.parametrize("via,ext", LOADS)
def test_a_host_load_never_asks_for_the_gpu_apis(monkeypatch, tmp_path, setter, via, ext):
    loaded = []
    ld = _loader(monkeypatch, tmp_path, loaded, ext)
    _torch_at(monkeypatch, "2.5.1", setter=False, host_cache=False)
    assert _load(ld, via, "cpu") == {"w": "cpu"} and setter == []


def test_without_package_metadata_the_refusal_names_what_is_missing_never_a_guessed_release(monkeypatch, setter):
    def absent(name):
        raise importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(importlib.metadata, "requires", absent)
    _torch_at(monkeypatch, "2.11.0", host_cache=False)
    with pytest.raises(RuntimeError, match=r"2\.11\.0 lacks; this python has no package metadata for neurobrix .*"
                       r"pip install -e \"\.\[pytorch\]\""):
        TA.configure_torch_allocator(_plan(V100))


def test_a_torch_with_both_apis_applies_the_declared_setting(monkeypatch, setter):
    _torch_at(monkeypatch, "2.12.0")
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
