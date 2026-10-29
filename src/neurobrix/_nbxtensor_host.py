"""The engine provides nbxtensor with what the library does not own (`nbxtensor.host`): its vendor files, its installation
refusals and the Triton DtypeEngine's operand decisions; it installs its census as the library's launch observer and
miss explainer.
Two of its settings go through the library's public API instead, at install: its certified directory
(`ops.autotune_settings.set_store`) and, when a rung is imposed on the process (`NBX_PRISM_BUDGET_MB`), the memory limit
(`nbxtensor.memory.set_limit`, through `core.prism.memory_budget.apply_imposed_rung`).

Every service resolves its engine function AT CALL TIME, in the module that owns it, so a value or a function the
engine changes later — a test's monkeypatch included — is the one read. Installed once by `neurobrix/__init__.py`.
Torch-free (R33).
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path


def _call(module: str, attr: str):
    def service(*args, **kwargs):
        return getattr(importlib.import_module(module), attr)(*args, **kwargs)
    service.__qualname__ = f"{module}.{attr}"
    return service


def _config_dir(name: str):
    def service() -> Path:
        return Path(importlib.import_module("neurobrix.core.config.loader").CONFIG_ROOT) / name
    service.__qualname__ = f"CONFIG_ROOT/{name}"
    return service


class _CensusObserver:
    """The engine's census (`neurobrix.kernels.census`) as the library's launch observer, resolved at call time: the
    census installs itself before any kernel is defined, and a test that replaces it is the one consulted."""

    @staticmethod
    def active() -> bool:
        return importlib.import_module("neurobrix.kernels.census").active()

    @staticmethod
    def record(tuned, key) -> None:
        return importlib.import_module("neurobrix.kernels.census").record(tuned, key)

    @staticmethod
    def shadow_run(*args, **kwargs):
        return importlib.import_module("neurobrix.kernels.census").shadow_run(*args, **kwargs)


def install() -> None:
    from nbxtensor.host import provide
    loader, dtype = "neurobrix.core.config.loader", "neurobrix.triton.dtype"
    provide("vendor_config", _call(loader, "get_vendor_config"))
    provide("vendor_param", _call(loader, "vendor_param"))
    provide("vendors_dir", _config_dir("vendors"))
    from nbxtensor.ops.autotune_settings import set_store, set_miss_explainer
    set_store(_config_dir("autotune"))          # the engine's committed certified directory, resolved when asked
    # the certification census: the library's launch observer (a dry run that records every key) and the account of a
    # confirmation run's miss (the census row that should hold it) — both the library's optional hooks, read at call time
    from nbxtensor.ops.launch_observer import set_launch_observer
    set_launch_observer(_CensusObserver())
    set_miss_explainer(_call("neurobrix.kernels.census_table", "census_row"))
    # A rung imposed on this process holds the Triton allocator to that card, through the library's public limit;
    # without one the library's default (the physical card) is already in force, and the allocator is not imported.
    # (the variable is core.prism.memory_budget._door_mb's; read here first so a process without a rung imports neither
    # the engine's planner nor the allocator at `import neurobrix` — the agent package's purity, tests/unit/agent)
    if os.environ.get("NBX_PRISM_BUDGET_MB"):
        importlib.import_module("neurobrix.core.prism.memory_budget").apply_imposed_rung()
