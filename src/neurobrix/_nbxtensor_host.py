"""The engine provides nbxtensor with what the library does not own (`nbxtensor.host`): its vendor files, its drain
policy, its memory classes, its installation refusals, its launch census and the Triton DtypeEngine's operand decisions.
Two of its settings go through the library's public API instead, at install: its certified directory
(`ops.autotune_certified.set_store`) and, when a rung is imposed on the process (`NBX_PRISM_BUDGET_MB`), the memory limit
(`nbxtensor.memory.set_limit`, through `core.prism.memory_budget.apply_imposed_rung`).

Every service resolves its engine function AT CALL TIME, in the module that owns it, so a value or a function the
engine changes later — a test's monkeypatch included — is the one read. Installed once by `neurobrix/__init__.py`.
Torch-free (R33).
"""
from __future__ import annotations

import importlib
from pathlib import Path


def _call(module: str, attr: str):
    def service(*args, **kwargs):
        return getattr(importlib.import_module(module), attr)(*args, **kwargs)
    service.__qualname__ = f"{module}.{attr}"
    return service


def _module(module: str):
    def service():
        return importlib.import_module(module)
    service.__qualname__ = module
    return service


def _config_dir(name: str):
    def service() -> Path:
        return Path(importlib.import_module("neurobrix.core.config.loader").CONFIG_ROOT) / name
    service.__qualname__ = f"CONFIG_ROOT/{name}"
    return service


def install() -> None:
    from nbxtensor.host import provide
    loader, dtype = "neurobrix.core.config.loader", "neurobrix.triton.dtype"
    provide("vendor_config", _call(loader, "get_vendor_config"))
    provide("vendor_param", _call(loader, "vendor_param"))
    provide("vendors_dir", _config_dir("vendors"))
    from nbxtensor.ops.autotune_certified import set_store
    set_store(_config_dir("autotune")())        # the engine's committed certified directory, through the public API
    provide("deferred_drain_policy_of", _call("neurobrix.core.prism.memory_budget", "deferred_drain_policy_of"))
    provide("memory_class_gb", _call("neurobrix.core.prism.structure", "memory_class_gb"))
    provide("memory_class_from_profile_id", _call("neurobrix.core.prism.structure", "memory_class_from_profile_id"))
    provide("installation_refusals", _call("neurobrix.core.paths", "installation_refusals"))
    provide("census", _module("neurobrix.kernels.census"))
    provide("census_table", _module("neurobrix.kernels.census_table"))
    for name in ("matrix_unit_representation", "matrix_unit_flash_representation", "attention_operand_dtypes",
                 "attention_operand_alignment", "matrix_unit_split_scale"):
        provide(name, _call(dtype, name))
    # A rung imposed on this process holds the Triton allocator to that card, through the library's public limit;
    # without one the library's default (the physical card) is already in force, and the allocator is not imported.
    budget = importlib.import_module("neurobrix.core.prism.memory_budget")
    if budget._door_mb() is not None:
        budget.apply_imposed_rung()
