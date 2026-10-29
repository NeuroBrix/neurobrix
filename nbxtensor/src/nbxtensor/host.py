"""What the application that runs nbxtensor provides to it: the facts and policies the library does not own.

nbxtensor computes; the application decides. Some decisions the kernels and the allocator need are not tensor facts:
the hardware vendor's files (tiles, budgets, the matrix unit) and where they live, the store of certified kernel
settings, how much host memory is free, the memory class a card belongs to, a rung imposed on the allocator, the
dtype policy that picks an attention's operand dtypes, the census that records launches without running them. The
application provides each of them ONCE, by name (`provide`); the library reads them by name (`need`). A service no
application provided is refused by name, never guessed.

Each service is a callable. An application provides a function that reads its own state at call time, so a value it
changes later is the one read.

Torch-free, numpy-free, and nothing here imports the application.
"""
from __future__ import annotations

from typing import Callable, Dict

#: the services this library reads, and what each one answers — the contract an application fulfils
SERVICES: Dict[str, str] = {
    "vendor_config": "(vendor, architecture) -> the vendor file of that architecture, as a dict",
    "vendor_param": "(section, key, default) -> one value of the executing device's vendor file",
    "vendors_dir": "() -> the directory of the vendor files (pathlib.Path)",
    "tiling_policy": "() -> the split sizes the tiling rules read, as a dict",
    "certified_dir": "() -> the directory of the certified kernel settings (pathlib.Path)",
    "memory_state": "() -> the host memory reading (available_mb, swap_used_mb, ...)",
    "host_shares_memory_with_device": "(device) -> True when host and device draw on one pool",
    "imposed_rung_mb": "() -> the memory rung imposed on the allocator in MB, or None",
    "deferred_drain_policy_of": "(memory section of a vendor file) -> the deferred-free drain policy",
    "memory_class_gb": "(memory_mb) -> the memory class of a card, in GB",
    "memory_class_from_profile_id": "(hardware profile id) -> its memory class in GB, or None",
    "installation_refusals": "(path) -> why an installation path may not hold a backend build; optional: absent, none is refused",
    "census": "() -> the launch census in force (active, record, shadow_run); optional: absent, none is active",
    "census_table": "() -> the census table reader",
    "canonical_aten": "(op name) -> the canonical ATen op name",
    "matrix_unit_representation": "(...) -> how the matrix unit carries one GEMM's operands",
    "matrix_unit_flash_representation": "(...) -> how the matrix unit carries one attention's operands",
    "attention_operand_dtypes": "(...) -> the operand dtypes an attention runs at",
    "attention_operand_alignment": "(...) -> the alignment an attention's operands are given",
    "matrix_unit_split_scale": "(...) -> the scale of a split fp32 operand on the matrix unit",
    "profile_triton_has_fp64": "() -> True when the profile in force computes float64",
}

class _NoCensus:
    """The census an application without one runs under: never active, it records nothing and holds no shadow memory.
    A census is an observer of launches, not a fact a computation reads: with none, there is nothing to observe."""

    @staticmethod
    def active() -> bool:
        return False

    @staticmethod
    def record(*args, **kwargs):
        raise RuntimeError("nbxtensor.host: a launch was recorded with no census provided — record() is reached only "
                           "under an active census")


#: hooks an application MAY provide; absent, the library runs as the value here describes (never a guessed fact)
def _no_refusal(path) -> list:
    """The installation rule of an application without one: no path is refused."""
    return []


_ABSENT = {"census": lambda: _NoCensus,
           # an application's rule on where a backend build may live (NeuroBrix: off its workshop's cleared storage);
           # absent, the library refuses no installation
           "installation_refusals": _no_refusal}

_PROVIDED: Dict[str, Callable] = {}


def provide(name: str, fn: Callable) -> None:
    """The application provides service `name` (one of `SERVICES`). Providing it again replaces it."""
    if name not in SERVICES:
        raise KeyError(f"nbxtensor.host: {name!r} is not a service this library reads (known: {', '.join(sorted(SERVICES))})")
    if not callable(fn):
        raise TypeError(f"nbxtensor.host: service {name!r} must be callable, got {type(fn).__name__}")
    _PROVIDED[name] = fn


def need(name: str) -> Callable:
    """Service `name` as the application provided it; refused by name when none did."""
    try:
        return _PROVIDED[name]
    except KeyError:
        if name in _ABSENT:
            return _ABSENT[name]
        if name not in SERVICES:
            raise KeyError(f"nbxtensor.host: {name!r} is not a service this library reads") from None
        raise RuntimeError(f"ZERO FALLBACK: nbxtensor needs the service {name!r} ({SERVICES[name]}) and the application "
                           f"running it provided none (nbxtensor.host.provide)") from None


def provided() -> Dict[str, Callable]:
    """The services provided so far (a copy)."""
    return dict(_PROVIDED)
