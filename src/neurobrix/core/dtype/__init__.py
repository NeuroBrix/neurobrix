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
    DTYPE_MAP,
    DTYPE_TO_STR,
    BYTES_MAP,
    HARDWARE_DTYPE_SUPPORT,
    get_dtype_bytes,
    get_torch_dtype,
    dtype_to_str,
    architecture_supports_dtype,
)

# Converter - safe conversion functions
from neurobrix.core.dtype.converter import (
    safe_dtype_convert,
    safe_dtype_convert_dict,
    calculate_dtype_multiplier,
    resolve_safe_fallback,
)

# DtypeEngine - single entry point for all dtype decisions
from neurobrix.core.dtype.engine import DtypeEngine

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
    # Converter
    "safe_dtype_convert",
    "safe_dtype_convert_dict",
    "calculate_dtype_multiplier",
    "resolve_safe_fallback",
    # DtypeEngine
    "DtypeEngine",
]
