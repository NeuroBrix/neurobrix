"""
Unified Kernel Resolver - NeuroBrix.

Combines ATen classification with cascade kernel resolution.
Single entry point for all kernel lookups.

Resolution Flow:
1. Classify op (TRITON/METADATA)
2. Map to kernel file name
3. Cascade through tiers: arch-specific → common
4. Return kernel function or raise if not found

ZERO FALLBACK: Missing kernel = explicit error with guidance.
"""

import os
import sys
import importlib
from pathlib import Path
from typing import Callable, Optional, List, Dict, Any, Union

from .registry import KERNEL_REGISTRY
from .classification import OpExecution, get_execution_type, ATEN_CLASSIFICATION
from .mapping import get_kernel_op_name, ATEN_TO_KERNEL
from .exceptions import KernelNotFoundError


# =============================================================================
# PERFORMANCE CACHES - Critical for runtime speed
# =============================================================================

# Cache: (family, vendor, arch, kernel_op) -> module
_MODULE_CACHE: Dict[tuple, Any] = {}

# Cache: op_type_full -> (execution_type, kernel_op_name)
_OP_INFO_CACHE: Dict[str, tuple] = {}


# =============================================================================
# Architecture → Tier Cascade Mapping
# =============================================================================

NVIDIA_TIER_CASCADE = {
    "hopper": ["hopper", "ampere", "volta", "common"],
    "ada": ["ada", "ampere", "volta", "common"],
    "ampere": ["ampere", "volta", "common"],
    "turing": ["turing", "volta", "common"],
    "volta": ["volta", "common"],
    "common": ["common"],
}

AMD_TIER_CASCADE = {
    "cdna3": ["cdna3", "cdna2", "cdna", "common"],
    "cdna2": ["cdna2", "cdna", "common"],
    "cdna": ["cdna", "common"],
    "rdna3": ["rdna3", "rdna2", "common"],
    "rdna2": ["rdna2", "common"],
    "common": ["common"],
}


def _get_tier_cascade(vendor: str, arch: str) -> List[str]:
    """Get tier cascade for vendor/architecture."""
    if vendor == "nvidia":
        return NVIDIA_TIER_CASCADE.get(arch.lower(), ["common"])
    elif vendor == "amd":
        return AMD_TIER_CASCADE.get(arch.lower(), ["common"])
    else:
        return ["common"]


# =============================================================================
# Auto-Discovery (import all kernel modules to trigger @register_kernel)
# =============================================================================

_kernels_loaded = False


def _ensure_kernels_loaded():
    """Import all kernel modules to trigger @register_kernel decorators."""
    global _kernels_loaded
    if _kernels_loaded:
        return

    kernels_dir = Path(__file__).parent
    ops_dir = kernels_dir / "ops"

    if not ops_dir.exists():
        return

    for root, dirs, files in os.walk(ops_dir):
        # Skip __pycache__
        if "__pycache__" in dirs:
            dirs.remove("__pycache__")

        for file in files:
            if file.endswith(".py") and not file.startswith("_"):
                file_path = Path(root) / file
                try:
                    # Convert path to module: kernels/ops/image/nvidia/common/add.py
                    # → kernels.ops.image.nvidia.common.add
                    rel_path = file_path.relative_to(kernels_dir)
                    module_parts = list(rel_path.with_suffix("").parts)
                    module_name = "neurobrix.kernels." + ".".join(module_parts)
                    importlib.import_module(module_name)
                except Exception:
                    # Log but dont crash - maybe syntax error in one kernel
                    pass

    _kernels_loaded = True


# =============================================================================
# Public API
# =============================================================================

def get_kernel(
    op_type: str,
    family: str = "image",
    vendor: str = "nvidia",
    arch: str = "volta",
    hw: Dict[str, str] = None,
) -> Optional[Callable]:
    """
    Get kernel function for an ATen operation.

    Args:
        op_type: ATen op name (e.g., "aten::addmm" or "addmm")
        family: Model family (image, llm, video, audio)
        vendor: Hardware vendor (nvidia, amd)
        arch: Architecture (volta, ampere, hopper, etc.)
        hw: Optional hardware dict with vendor/architecture keys

    Returns:
        Kernel function or None if metadata op

    Raises:
        KernelNotFoundError: If op unknown or kernel not found (ZERO FALLBACK)
    """
    _ensure_kernels_loaded()

    # Extract from hw dict if provided
    if hw is not None:
        vendor = hw.get("vendor", vendor)
        arch = hw.get("architecture", arch)

    # Normalize op name
    if not op_type.startswith("aten::"):
        op_type_full = f"aten::{op_type}"
    else:
        op_type_full = op_type

    # 1. Check if known ATen op
    if op_type_full in ATEN_CLASSIFICATION:
        # Classify operation
        exec_type = get_execution_type(op_type_full)

        # Metadata ops - no kernel needed
        if exec_type == OpExecution.METADATA:
            return None

        # Get kernel operation name
        kernel_op = get_kernel_op_name(op_type_full)
        if kernel_op is None:
            return None
    else:
        # Direct kernel name (not ATen op)
        kernel_op = op_type.replace("aten::", "")

    # 2. Try registry with cascade
    # Return PURE KERNEL ONLY - adapter handles all translation
    tiers = _get_tier_cascade(vendor, arch)
    for tier in tiers:
        key = (family, vendor, tier, kernel_op)
        if key in KERNEL_REGISTRY:
            return KERNEL_REGISTRY[key].impl

    # 3. Try dynamic import with cascade (for auto-registration)
    for tier in tiers:
        try:
            module_path = f"neurobrix.kernels.ops.{family}.{vendor}.{tier}.{kernel_op}"
            importlib.import_module(module_path)

            # After import, check registry again
            key = (family, vendor, tier, kernel_op)
            if key in KERNEL_REGISTRY:
                return KERNEL_REGISTRY[key].impl

        except ImportError:
            continue

    # 4. ZERO FALLBACK - explicit error
    available_ops = _list_available_ops(family, vendor)
    tiers_str = " → ".join(tiers)

    raise KernelNotFoundError(
        f"No kernel found for {op_type_full}\n"
        f"Context: {family}/{vendor}/{arch}\n"
        f"Tiers searched: {tiers_str}"
    )


def get_kernel_module(
    op_type: str,
    family: str = "image",
    vendor: str = "nvidia",
    arch: str = "volta",
) -> Optional[Any]:
    """
    Get the full kernel module for an op.
    Used by adapter to access JIT functions directly.

    PERFORMANCE: Uses aggressive caching to minimize per-op overhead.
    """
    # Fast path: check op info cache first
    if not op_type.startswith("aten::"):
        op_type_full = f"aten::{op_type}"
    else:
        op_type_full = op_type

    # Get kernel op name (cached lookup)
    if op_type_full in _OP_INFO_CACHE:
        _, kernel_op = _OP_INFO_CACHE[op_type_full]
    else:
        kernel_op = get_kernel_op_name(op_type_full)
        if kernel_op is not None:
            kernel_op = kernel_op.replace("aten::", "")
        exec_type = get_execution_type(op_type_full) if op_type_full in ATEN_CLASSIFICATION else None
        _OP_INFO_CACHE[op_type_full] = (exec_type, kernel_op)

    if kernel_op is None:
        return None

    # Check module cache (fastest path)
    cache_key = (family, vendor, arch, kernel_op)
    if cache_key in _MODULE_CACHE:
        return _MODULE_CACHE[cache_key]

    # Cascade through tiers
    tiers = _get_tier_cascade(vendor, arch)
    for tier in tiers:
        module_path = f"neurobrix.kernels.ops.{family}.{vendor}.{tier}.{kernel_op}"

        # Check sys.modules first
        if module_path in sys.modules:
            module = sys.modules[module_path]
            _MODULE_CACHE[cache_key] = module
            return module

        # Try import
        try:
            module = importlib.import_module(module_path)
            _MODULE_CACHE[cache_key] = module
            return module
        except ImportError:
            continue

    # Cache None to avoid repeated failed lookups
    _MODULE_CACHE[cache_key] = None
    return None


def run_op(
    op_type: str,
    inputs: List,
    outputs: List = None,
    attrs: Dict = None,
    family: str = "image",
    vendor: str = "nvidia",
    arch: str = "volta",
    hw: Dict[str, str] = None,
) -> Any:
    """
    Resolve and execute kernel in one call.

    For METADATA ops, returns None (caller handles via PyTorch).
    """
    kernel = get_kernel(op_type, family, vendor, arch, hw)

    if kernel is None:
        # Metadata op - caller should handle
        return None

    # Support both old (inputs, outputs, attrs) and new (*tensors) calling conventions
    if outputs is not None or attrs is not None:
        return kernel(inputs, outputs or [], attrs or {})
    else:
        # inputs is actually a list of tensors
        return kernel(*inputs)


def list_kernels(
    family: str = None,
    vendor: str = None,
    tier: str = None,
) -> List[tuple]:
    """List all registered kernels with optional filters."""
    _ensure_kernels_loaded()

    keys = list(KERNEL_REGISTRY.keys())

    if family:
        keys = [k for k in keys if k[0] == family]
    if vendor:
        keys = [k for k in keys if k[1] == vendor]
    if tier:
        keys = [k for k in keys if k[2] == tier]

    return keys


def _list_available_ops(family: str, vendor: str) -> List[str]:
    """List available ops for a family/vendor."""
    keys = list_kernels(family=family, vendor=vendor)
    ops = sorted(set(k[3] for k in keys))
    return ops if ops else ["none"]
