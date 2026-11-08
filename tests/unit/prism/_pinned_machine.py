"""The machine a Prism gate plans for, BUILT by the gate — never found on the machine it runs on.

Shared by every prism gate that plans a real container. It exists because the first versions
of these gates found three things on the rack and silently needed them everywhere:

* the CACHE, written as a path in each test (`os.path.expanduser("~/.neurobrix/" ...)`, the word
  split in two so a text search would not see it). The engine resolves its cache through one
  door, `neurobrix.core.paths.cache_dir()` — `NEUROBRIX_CACHE`, then `~/.neurobrix/paths.json`,
  then the default — and on the Mac the models live on the mount that door names, not at the
  literal. Every cell skipped there, in silence;
* a MISSING container answered with an anonymous `pytest.skip`, or (2026-09-24 to 2026-10-12) with a
  FAIL — which on the M4 made 140 failures + 14 errors of containers that were simply not installed,
  and a real red drowned in them. It is now `tests.conftest.installed_container`, the suite's one
  answer: a skip that NAMES the container and the path the engine's door resolved, so the absence
  is read in the summary and never counted as a pass or as a defect;
* HARDWARE PROFILES read by id from `config/hardware/default-*.yml`, which are generated per
  machine and gitignored. The rack's `default-ff6008b7` does not exist on the Mac, and neither
  profile exists on a third machine. Here each profile is written from the values it had when
  the scenario was measured and loaded through the engine's own loader, so the gate carries the
  machine it plans for (vacuous-gates register 102).

What remains the machine's own: the containers themselves. A gate about a real container's
graph cannot be built without it; it says so BY NAME when it is absent.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

import nbxtensor.backend.host_memory as host_memory
import neurobrix.core.prism.loader as profile_loader
import neurobrix.core.prism.solver as solver_mod
from nbxtensor.backend.host_memory import MemoryState
from neurobrix.core.prism.memory_budget import DeviceReading
from tests.conftest import installed_container


# ─────────────────────────────── the containers ───────────────────────────────

def container_root(model: str) -> Path:
    """The container's directory in the cache the ENGINE resolves; a named skip when absent."""
    return installed_container(model)


# ─────────────────────────────── the profiles ───────────────────────────────
# Each is the profile as it was generated on the machine the scenario was measured on, reduced to
# the fields the loader reads. Named by what they are, not by a machine-local id.

def _vendor_measured(vendor: str) -> dict:
    """The `measured:` block of a tracked vendor file, as the generated profile of that machine carries it."""
    import yaml
    from pathlib import Path
    f = Path(__file__).resolve().parents[3] / "src" / "neurobrix" / "config" / "vendors" / f"{vendor}.yml"
    return dict((yaml.safe_load(f.read_text()) or {}).get("measured") or {})


APPLE_M4_PRO = {        # the Mac's `default-9f169c79`, 2026-09-24; its engine base as remeasured 2026-10-29 01:47
    "id": "scenario-apple-m4-pro-18g", "vendor": "apple", "preferred_dtype": "bfloat16",
    "cpu": {"io_workers": 8, "model": "Apple M4 Pro", "cores": 12, "threads": 12, "ram_mb": 24576,
            "architecture": "arm64", "features": [],
            # what the engine's device work adds to a process that has planned, measured on the Mac by
            # autodetect._measure_runtime_base_mb (its default-9f169c79 regenerated 2026-10-29 01:47, the Mac 21:32):
            # without it the replica priced the base at 0 and admitted Janus-Pro-7B whole at 16 732 MB free, where
            # the Mac's own door, 254 MB dearer, streamed (2026-10-29 21:22)
            "runtime_base_mb": {"compiled": 184, "triton": 254}},
    "devices": [{"index": 0, "brand": "apple", "model": "Apple M4 Pro", "memory_mb": 18186,
                 "compute_capability": "0.0", "supports_dtypes": ["float32", "float16", "bfloat16"],
                 "architecture": "apple_silicon", "pcie_version": "N/A", "unified_memory": True,
                 "host_memory_mb": 24576,
                 # MTLDevice.maxBufferLength, what autodetect writes on every Apple device (13 639 MB, 2026-10-04)
                 "max_allocation_mb": 13639,
                 # the Mac's generated profile holds its device's measured facts WHOLE (a profile's block replaces the
                 # vendor file's), with the arena factor the engine measures there since 2026-11-03 (HAT-L x4 triton,
                 # the allocator's live peak over the walk, 2026-10-28)
                 "measured": {**_vendor_measured("apple/apple_m4_pro"), "triton_arena_activation_factor": 1.19}}],
    # the per-piece reload the engine measures at the first launch since 1604166f (no longer in the vendor file): the
    # triton BF16 rate it measured on Janus-Pro-7B's segments (2.1 GB/s, 2026-10-27), over the vendor's host facts
    "host_measured": {"weight_reload_gbps": {"triton": {"BF16": 2.1}}},
    "interconnect": {"groups": []},
    "pcie_fallback": {"version": "N/A", "lanes": 16, "bandwidth_gbps": 32},
}

V100_16GB = {           # the rack's `default-ff6008b7`, one V100-SXM2-16GB, 2026-09-24
    "id": "scenario-v100-16gb", "vendor": "dell", "preferred_dtype": "float16",
    "cpu": {"io_workers": 8, "model": "Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz", "cores": 40, "threads": 80,
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
    """The host's memory reading, as the scenario measured it — the planning process's own footprint with it.

    The scenario's available figure was read with the planning process already inside it, so what that process
    holds is part of the injected reading: nothing beyond it (`host_footprint.process_footprint_now`, the resident
    term of the plan's host footprint). Left live, it is whatever the suite ran before: mochi-1-preview on the calm
    Mac streamed its VAE alone (56 MB resident) and sliced it after DeepSeek-Coder-V2-Lite's fused graphs had been
    normalized in the same process (3 400 MB resident, 2026-11-03)."""
    from neurobrix.core.prism import host_footprint
    st = MemoryState(total_mb=total_mb, available_mb=available_mb, source=f"injected: {why}")
    monkeypatch.setattr(solver_mod, "memory_state", lambda: st)
    monkeypatch.setattr(host_memory, "memory_state", lambda: st)
    monkeypatch.setattr(host_footprint, "process_footprint_now", lambda: 0)


def pin_reload(monkeypatch, engine: str, stored: str, gbps: float, why: str) -> None:
    """The host's per-token reload rate for `engine` at `stored` width (`weight_reload_gbps`, which Prism prices a
    streamed piece's reload at and refuses to estimate without — e0d521e8), as the scenario states it, with `why`."""
    import dataclasses
    from neurobrix.core.prism import time_estimate as TE
    real = TE.host_facts

    def facts(*a, **k):
        f = real(*a, **k)
        table = {e: dict(v) for e, v in (f.reload_gbps or {}).items()}
        table.setdefault(engine, {})[stored] = gbps
        return dataclasses.replace(f, reload_gbps=table)
    facts.why = why
    monkeypatch.setattr(TE, "host_facts", facts)


def pin_dedicated_card(monkeypatch, driver_total_mb: int, own_context_mb: int, why: str) -> None:
    """Every discrete card reads as dedicated: nothing else on it, the runtime's own context.

    BOTH readings `_prepare_devices` takes of a discrete card are pinned — the sharing facts and
    the driver's free figure — so a machine whose own driver answers differently (a Mac has no
    CUDA card at index 0; a rack card may be busy) plans the same card."""
    from nbxtensor.memory.allocator import DeviceAllocator
    free = driver_total_mb - own_context_mb
    monkeypatch.setattr(solver_mod, "read_device_sharing", lambda _index: DeviceReading(
        kind="device", capacity_mb=driver_total_mb, free_mb=free,
        own_context_mb=own_context_mb, measured=True, source=f"injected: {why}"))
    monkeypatch.setattr(DeviceAllocator, "free_memory_mb", staticmethod(lambda _index: free))


def pin_no_calibration(monkeypatch) -> None:
    """The engine's calibration store (`~/.neurobrix/calibration`, `core/dtype/calibration.STORE_ROOT`) read as EMPTY:
    a record there narrows the compiled branch's precision contract at plan time (`runtime_widths.plan_time_contract`)
    and moves a plan by the machine it runs on — DeepSeek-Coder-V2-Lite's compiled lm_head priced 886 MB on the rack
    (a record in its store) and 1 772 MB on the Mac (none), the same test on the same container (2026-10-28). The
    records an ARTIFACT carries (its profile.json) still apply: they travel with the container."""
    import neurobrix.core.dtype.calibration as calibration
    monkeypatch.setattr(calibration, "STORE_ROOT", Path("/nonexistent-calibration-store"))


def impose_rung(monkeypatch, rung_mb) -> None:
    monkeypatch.setenv("NBX_PRISM_BUDGET_MB", str(int(rung_mb)))


def no_door(monkeypatch) -> None:
    monkeypatch.delenv("NBX_PRISM_BUDGET_MB", raising=False)


def pin_shared_card(monkeypatch, driver_total_mb: int, held_by_others_mb: int, own_context_mb: int,
                    why: str) -> None:
    """Every discrete card reads as SHARED: another process holds `held_by_others_mb` of it, so the
    law rounds the free reading down onto the ladder and the rung sits below the capacity."""
    from nbxtensor.memory.allocator import DeviceAllocator
    free = driver_total_mb - own_context_mb - held_by_others_mb
    monkeypatch.setattr(solver_mod, "read_device_sharing", lambda _index: DeviceReading(
        kind="device", capacity_mb=driver_total_mb, free_mb=free, held_by_others_mb=held_by_others_mb,
        own_context_mb=own_context_mb, measured=True, source=f"injected: {why}"))
    monkeypatch.setattr(DeviceAllocator, "free_memory_mb", staticmethod(lambda _index: free))
