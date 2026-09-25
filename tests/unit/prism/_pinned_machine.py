"""The machine a Prism gate plans for, BUILT by the gate — never found on the machine it runs on.

Shared by every prism gate that plans a real container. It exists because the first versions
of these gates found three things on the rack and silently needed them everywhere:

* the CACHE, written as a path in each test (`os.path.expanduser("~/.neurobrix/" ...)`, the word
  split in two so a text search would not see it). The engine resolves its cache through one
  door, `neurobrix.core.paths.cache_dir()` — `NEUROBRIX_CACHE`, then `~/.neurobrix/paths.json`,
  then the default — and on the Mac the models live on the mount that door names, not at the
  literal. Every cell skipped there, in silence;
* a MISSING container answered with `pytest.skip`, so a machine without the model reported a
  gate that proved nothing as a gate that did not fail. Here it FAILS, naming the path and who
  configured it: a gate that cannot run its scenario has not passed;
* HARDWARE PROFILES read by id from `config/hardware/default-*.yml`, which are generated per
  machine and gitignored. The rack's `default-ff6008b7` does not exist on the Mac, and neither
  profile exists on a third machine. Here each profile is written from the values it had when
  the scenario was measured and loaded through the engine's own loader, so the gate carries the
  machine it plans for (vacuous-gates register 102).

What remains the machine's own: the containers themselves. A gate about a real container's
graph cannot be built without it; it must say so loudly when it is absent.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

import neurobrix.core.host_memory as host_memory
import neurobrix.core.prism.loader as profile_loader
import neurobrix.core.prism.solver as solver_mod
from neurobrix.core import paths
from neurobrix.core.host_memory import MemoryState
from neurobrix.core.prism.memory_budget import DeviceReading


# ─────────────────────────────── the containers ───────────────────────────────

def container_root(model: str) -> Path:
    """The container's directory in the cache the ENGINE resolves; a failure when absent."""
    root = paths.cache_dir() / model
    if not (root / "components").is_dir():
        said_by = paths.describe().get("cache", {}).get("said_by", "?")
        pytest.fail(
            f"{model} is not in this machine's cache ({paths.cache_dir()}, said by {said_by}). "
            f"This gate plans that container and cannot run without it — point NEUROBRIX_CACHE "
            f"or ~/.neurobrix/paths.json at a cache that holds it. A skip here would report a gate "
            f"that proved nothing as one that did not fail.", pytrace=False)
    return root


# ─────────────────────────────── the profiles ───────────────────────────────
# Each is the profile as it was generated on the machine the scenario was measured on, reduced to
# the fields the loader reads. Named by what they are, not by a machine-local id.

APPLE_M4_PRO = {        # the Mac's `default-9f169c79`, 2026-09-24
    "id": "scenario-apple-m4-pro-18g", "vendor": "apple", "preferred_dtype": "bfloat16",
    "cpu": {"model": "Apple M4 Pro", "cores": 12, "threads": 12, "ram_mb": 24576,
            "architecture": "arm64", "features": []},
    "devices": [{"index": 0, "brand": "apple", "model": "Apple M4 Pro", "memory_mb": 18186,
                 "compute_capability": "0.0", "supports_dtypes": ["float32", "float16", "bfloat16"],
                 "architecture": "apple_silicon", "pcie_version": "N/A", "unified_memory": True,
                 "host_memory_mb": 24576}],
    "interconnect": {"groups": []},
    "pcie_fallback": {"version": "N/A", "lanes": 16, "bandwidth_gbps": 32},
}

V100_16GB = {           # the rack's `default-ff6008b7`, one V100-SXM2-16GB, 2026-09-24
    "id": "scenario-v100-16gb", "vendor": "dell", "preferred_dtype": "float16",
    "cpu": {"model": "Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz", "cores": 40, "threads": 80,
            "ram_mb": 257530, "architecture": "x86_64", "features": ["avx2", "avx512f"]},
    "devices": [{"index": 0, "brand": "nvidia", "model": "Tesla V100-SXM2-16GB", "memory_mb": 16384,
                 "compute_capability": "7.0", "supports_dtypes": ["float32", "float16"],
                 "architecture": "volta", "pcie_version": "3.0"}],
    "interconnect": {"groups": []},
    "pcie_fallback": {"version": "3.0", "lanes": 16, "bandwidth_gbps": 32},
}


def profile(spec: dict):
    """Load `spec` through the engine's own profile loader, from a file this call writes."""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / f"{spec['id']}.yml").write_text(yaml.safe_dump(spec))
        saved = profile_loader.HARDWARE_DIR
        profile_loader.HARDWARE_DIR = Path(d)
        try:
            return profile_loader.load_profile(spec["id"])
        finally:
            profile_loader.HARDWARE_DIR = saved


# ─────────────────────────────── the readings ───────────────────────────────

def pin_host(monkeypatch, total_mb: int, available_mb: int, why: str) -> None:
    """The host's memory reading, as the scenario measured it."""
    st = MemoryState(total_mb=total_mb, available_mb=available_mb, source=f"injected: {why}")
    monkeypatch.setattr(solver_mod, "memory_state", lambda: st)
    monkeypatch.setattr(host_memory, "memory_state", lambda: st)


def pin_dedicated_card(monkeypatch, driver_total_mb: int, own_context_mb: int, why: str) -> None:
    """Every discrete card reads as dedicated: nothing else on it, the runtime's own context.

    BOTH readings `_prepare_devices` takes of a discrete card are pinned — the sharing facts and
    the driver's free figure — so a machine whose own driver answers differently (a Mac has no
    CUDA card at index 0; a rack card may be busy) plans the same card."""
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    free = driver_total_mb - own_context_mb
    monkeypatch.setattr(solver_mod, "read_device_sharing", lambda _index: DeviceReading(
        kind="device", capacity_mb=driver_total_mb, free_mb=free,
        own_context_mb=own_context_mb, measured=True, source=f"injected: {why}"))
    monkeypatch.setattr(DeviceAllocator, "free_memory_mb", staticmethod(lambda _index: free))


def impose_rung(monkeypatch, rung_mb) -> None:
    monkeypatch.setenv("NBX_PRISM_BUDGET_MB", str(int(rung_mb)))


def no_door(monkeypatch) -> None:
    monkeypatch.delenv("NBX_PRISM_BUDGET_MB", raising=False)


def pin_shared_card(monkeypatch, driver_total_mb: int, held_by_others_mb: int, own_context_mb: int,
                    why: str) -> None:
    """Every discrete card reads as SHARED: another process holds `held_by_others_mb` of it, so the
    law rounds the free reading down onto the ladder and the rung sits below the capacity."""
    from neurobrix.kernels.nbx_tensor import DeviceAllocator
    free = driver_total_mb - own_context_mb - held_by_others_mb
    monkeypatch.setattr(solver_mod, "read_device_sharing", lambda _index: DeviceReading(
        kind="device", capacity_mb=driver_total_mb, free_mb=free, held_by_others_mb=held_by_others_mb,
        own_context_mb=own_context_mb, measured=True, source=f"injected: {why}"))
    monkeypatch.setattr(DeviceAllocator, "free_memory_mb", staticmethod(lambda _index: free))
