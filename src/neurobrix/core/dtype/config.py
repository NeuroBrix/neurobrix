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
from typing import Dict


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
