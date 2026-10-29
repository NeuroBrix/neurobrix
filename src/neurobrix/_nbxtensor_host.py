"""The engine provides nbxtensor with what the library does not own (`nbxtensor.host`): its vendor files and certified
store, its host memory reading, its imposed rung and drain policy, its memory classes, its installation refusals, its
launch census, its op classification and the Triton DtypeEngine's operand decisions.

Every service resolves its engine function AT CALL TIME, in the module that owns it: nothing here imports the engine at
install (no import cost, no import order), and a value or a function the engine changes later — a test's monkeypatch
included — is the one read. Installed once by `neurobrix/__init__.py`. Torch-free (R33).
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
    provide("device_arch_string", _call(loader, "device_arch_string"))
    provide("tiling_policy", _call(loader, "get_tiling_policy"))
    provide("vendors_dir", _config_dir("vendors"))
    provide("certified_dir", _config_dir("autotune"))
    provide("memory_state", _call("neurobrix.core.host_memory", "memory_state"))
    provide("host_shares_memory_with_device", _call("neurobrix.core.host_memory", "host_shares_memory_with_device"))
    provide("imposed_rung_mb", _call("neurobrix.core.prism.memory_budget", "_door_mb"))
    provide("deferred_drain_policy_of", _call("neurobrix.core.prism.memory_budget", "deferred_drain_policy_of"))
    provide("memory_class_gb", _call("neurobrix.core.prism.structure", "memory_class_gb"))
    provide("memory_class_from_profile_id", _call("neurobrix.core.prism.structure", "memory_class_from_profile_id"))
    provide("installation_refusals", _call("neurobrix.core.paths", "installation_refusals"))
    provide("census", _module("neurobrix.kernels.census"))
    provide("census_table", _module("neurobrix.kernels.census_table"))
    provide("canonical_aten", _call("neurobrix.kernels.classification", "canonical_aten"))
    for name in ("matrix_unit_representation", "matrix_unit_flash_representation", "attention_operand_dtypes",
                 "attention_operand_alignment", "matrix_unit_split_scale", "profile_triton_has_fp64"):
        provide(name, _call(dtype, name))
