# core/dtype/__init__.py
"""
NeuroBrix Dtype Module - Single Source of Truth

Consolidates all dtype-related code:
1. config.py - DTYPE_MAP, BYTES_MAP, HARDWARE_SUPPORT
2. converter.py - safe_dtype_convert(), calculate_dtype_multiplier()
3. engine.py - DtypeEngine (single dtype decision engine)

Previously duplicated across:
- core/prism/common/dtype_resolver.py
- core/prism/common/allocation.py
- core/runtime/weight_loader.py

ZERO HARDCODE: Import from here for all dtype operations.
"""

# Config - constants and mappings
from neurobrix.core.dtype.config import (
    BYTES_MAP,
    HARDWARE_DTYPE_SUPPORT,
    get_dtype_bytes,
    get_torch_dtype,
    dtype_to_str,
    architecture_supports_dtype,
    parse_dtype,
    strip_aten_prefix,
)

# Converter - safe conversion functions
from neurobrix.core.dtype.converter import (
    safe_dtype_convert,
    safe_dtype_convert_dict,
    calculate_dtype_multiplier,
    resolve_safe_fallback,
)

# DtypeEngine (the ATen branch's engine) and the torch dtype maps resolve
# on request: a --triton process imports this package for BYTES_MAP and the
# string helpers and must not load torch (R33).
_LAZY = {"DtypeEngine": "neurobrix.core.dtype.engine",
         "DTYPE_MAP": "neurobrix.core.dtype.config",
         "DTYPE_TO_STR": "neurobrix.core.dtype.config"}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(module), name)

__all__ = [
    # Config
    "DTYPE_MAP",
    "DTYPE_TO_STR",
    "BYTES_MAP",
    "HARDWARE_DTYPE_SUPPORT",
    "get_dtype_bytes",
    "get_torch_dtype",
    "dtype_to_str",
    "architecture_supports_dtype",
    "parse_dtype",
    "strip_aten_prefix",
    # Converter
    "safe_dtype_convert",
    "safe_dtype_convert_dict",
    "calculate_dtype_multiplier",
    "resolve_safe_fallback",
    # DtypeEngine
    "DtypeEngine",
]
