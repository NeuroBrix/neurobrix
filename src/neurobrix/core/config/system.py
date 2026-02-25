"""
NeuroBrix System Configuration

Universal constants for memory and dtype calculations.
These are mathematical constants, not configuration that varies.

NOTE: Dtype constants imported from neurobrix.core.dtype (single source of truth).
NO YAML FILE NEEDED - these values are universal facts.
"""

from typing import Dict, Any

# Import dtype constants from single source of truth
from neurobrix.core.dtype import BYTES_MAP as DTYPE_BYTES


# =============================================================================
# UNIVERSAL CONSTANTS
# =============================================================================

# Memory unit constants (mathematical facts)
BYTES_PER_KB = 1024
BYTES_PER_MB = 1024 * 1024
BYTES_PER_GB = 1024 * 1024 * 1024

# Prism solver defaults
PRISM_DEFAULTS = {
    "safety_margin": 0.95,  # Use 95% of VRAM capacity (tight for large models)
    "default_seq_len": 128,  # Conservative default; actual value from defaults.json
    "overhead_factor": 0.05,  # 5% fragmentation buffer (low for inference)
    # FGP (Fine-Grained Pipeline) settings
    "fgp_utilization_target": 0.85,  # Use 85% of GPU memory for FGP
    "fgp_max_blocks_per_stage": 7,   # Max transformer blocks per GPU (7 for 32GB, ~4 for 16GB)
    # Default values for model analysis (used when not specified in profile)
    "default_batch_size": 2,  # CFG batching (positive + negative prompts)
    "default_patch_size": 2,  # Standard DiT/Sana patch size
}

# Numerical Stability — precision control
PRECISION_DEFAULTS: Dict[str, Any] = {}


# =============================================================================
# ACCESSOR FUNCTIONS
# =============================================================================

def get_memory_constants() -> Dict[str, int]:
    """
    Get memory unit constants.

    Returns:
        {"bytes_per_kb": 1024, "bytes_per_mb": 1048576, "bytes_per_gb": 1073741824}
    """
    return {
        "bytes_per_kb": BYTES_PER_KB,
        "bytes_per_mb": BYTES_PER_MB,
        "bytes_per_gb": BYTES_PER_GB,
    }


def get_prism_defaults() -> Dict[str, Any]:
    """
    Get Prism solver defaults.

    Returns:
        {"safety_margin": 0.85, "default_seq_len": 4096}
    """
    return PRISM_DEFAULTS.copy()


def get_dtype_bytes() -> Dict[str, int]:
    """
    Get bytes per element for each dtype.

    Returns:
        {"float16": 2, "bfloat16": 2, "float32": 4, ...}
    """
    return DTYPE_BYTES.copy()


def get_precision_config() -> Dict[str, bool]:
    """
    Get numerical precision configuration.

    Returns:
        Precision configuration dict.
    """
    return PRECISION_DEFAULTS.copy()


def load_system_config() -> Dict[str, Any]:
    """
    Legacy function for backwards compatibility.

    Returns combined system config dict.
    """
    return {
        "memory": get_memory_constants(),
        "prism": get_prism_defaults(),
        "dtype_bytes": get_dtype_bytes(),
        "precision": get_precision_config(),
    }


# Convenience accessors
def bytes_to_mb(size_bytes: int) -> float:
    """Convert bytes to megabytes using config constant."""
    mem = get_memory_constants()
    return size_bytes / mem["bytes_per_mb"]


def bytes_to_gb(size_bytes: int) -> float:
    """Convert bytes to gigabytes using config constant."""
    mem = get_memory_constants()
    return size_bytes / mem["bytes_per_gb"]


def mb_to_bytes(size_mb: float) -> int:
    """Convert megabytes to bytes using config constant."""
    mem = get_memory_constants()
    return int(size_mb * mem["bytes_per_mb"])


def gb_to_bytes(size_gb: float) -> int:
    """Convert gigabytes to bytes using config constant."""
    mem = get_memory_constants()
    return int(size_gb * mem["bytes_per_gb"])
