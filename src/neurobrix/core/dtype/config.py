# core/dtype/config.py
"""
Dtype Configuration - Single Source of Truth

Consolidates dtype constants previously duplicated across:
- core/prism/common/dtype_resolver.py
- core/prism/common/allocation.py
- core/runtime/weight_loader.py

ZERO HARDCODE: All dtype-related constants defined here.
"""

import torch
from typing import Dict, Optional


# ============================================================================
# Dtype String to Torch Mapping
# ============================================================================

DTYPE_MAP: Dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}

# Reverse mapping: torch.dtype -> string
DTYPE_TO_STR: Dict[torch.dtype, str] = {v: k for k, v in DTYPE_MAP.items()}


# ============================================================================
# Bytes Per Element (for memory calculations)
# ============================================================================

BYTES_MAP: Dict[str, int] = {
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
}


def get_dtype_bytes(dtype_str: str) -> int:
    """
    Get bytes per element for a dtype string.

    Args:
        dtype_str: Dtype string (e.g., "float16", "bfloat16")

    Returns:
        Bytes per element (defaults to 4 if unknown)
    """
    return BYTES_MAP.get(dtype_str, 4)


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert dtype string to torch.dtype.

    Args:
        dtype_str: Dtype string (e.g., "float16", "bfloat16")

    Returns:
        torch.dtype (defaults to float32 if unknown)
    """
    return DTYPE_MAP.get(dtype_str, torch.float32)


def parse_dtype(dtype_str: str, compute_dtype: Optional[torch.dtype] = None) -> torch.dtype:
    """
    Parse dtype string to torch.dtype with optional Prism remap.

    Handles both "float16" and "torch.float16" formats.
    When compute_dtype is provided, remaps half-precision dtypes:
    bf16→fp16 when Prism wants fp16 (and vice versa).

    This is the SINGLE implementation of dtype parsing + Prism remap.
    All runtime code must use this instead of inline dicts or ad-hoc remaps.

    Args:
        dtype_str: Dtype string ("float16", "torch.float16", etc.)
        compute_dtype: Prism compute dtype for half-precision remap (optional)

    Returns:
        Resolved torch.dtype
    """
    # Strip "torch." prefix
    clean = dtype_str[6:] if dtype_str.startswith("torch.") else dtype_str
    parsed = DTYPE_MAP.get(clean, torch.float32)

    # Prism remap: bf16↔fp16 when hardware wants a different half-precision
    if compute_dtype is not None:
        if parsed == torch.bfloat16 and compute_dtype == torch.float16:
            return torch.float16
        if parsed == torch.float16 and compute_dtype == torch.bfloat16:
            return torch.bfloat16

    return parsed


def strip_aten_prefix(op_type: str) -> str:
    """
    Strip 'aten::' prefix and variant suffix from op_type string.

    "aten::_softmax" → "_softmax"
    "aten::mm"       → "mm"
    "custom::rms_norm" → "rms_norm"

    SINGLE implementation — all runtime code must use this.
    """
    # Strip namespace prefix (aten::, custom::, etc.)
    if "::" in op_type:
        op_name = op_type.split("::", 1)[1]
    else:
        op_name = op_type
    # Strip variant suffix (.default, .int, etc.)
    return op_name.split(".")[0]


def dtype_to_str(dtype: torch.dtype) -> str:
    """
    Convert torch.dtype to string.

    Args:
        dtype: torch.dtype

    Returns:
        Dtype string (defaults to "float32" if unknown)
    """
    return DTYPE_TO_STR.get(dtype, "float32")


# ============================================================================
# Hardware Support Mapping
# ============================================================================

# Which architectures support which dtypes
HARDWARE_DTYPE_SUPPORT: Dict[str, list] = {
    # NVIDIA
    "volta": ["float32", "float16"],  # V100
    "turing": ["float32", "float16"],  # RTX 20xx
    "ampere": ["float32", "float16", "bfloat16"],  # A100, RTX 30xx
    "hopper": ["float32", "float16", "bfloat16", "fp8"],  # H100

    # AMD
    "cdna": ["float32", "float16"],  # MI100
    "cdna2": ["float32", "float16", "bfloat16"],  # MI200
    "cdna3": ["float32", "float16", "bfloat16", "fp8"],  # MI300

    # CPU fallback
    "cpu": ["float32", "float64"],
}


def architecture_supports_dtype(arch: str, dtype_str: str) -> bool:
    """
    Check if a hardware architecture supports a dtype.

    Args:
        arch: Architecture string (e.g., "volta", "ampere")
        dtype_str: Dtype string (e.g., "bfloat16")

    Returns:
        True if supported, False otherwise
    """
    supported = HARDWARE_DTYPE_SUPPORT.get(arch.lower(), ["float32"])
    return dtype_str in supported
