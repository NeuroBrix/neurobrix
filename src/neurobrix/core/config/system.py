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
    # Driver/library overhead reserve (MB). It was subtracted from a single GPU's planning capacity
    # before any strategy parked the whole model on one device; see the READ TODAY note below for
    # where it applies now. Empirical derivation (P-PRISM-NEVER-REFUSE v2 B.4, 2026-05-12):
    # ~13 GiB of live NBX tensors produced a 16.6 GiB runtime peak on a
    # 32 GiB V100 — CUDA context + cuDNN/cuBLAS workspaces + Triton kernel
    # cache + caching-allocator fragmentation ≈ 3 GiB that no activation
    # estimator term covers. READ TODAY (2026-09-24) by the single-GPU KV-cache budget and the
    # stated plan margin only: the acceptance gates hold plans against the rung since the
    # 2026-09-21 memory law, which removed this reserve from them.
    "oom_reserve_mb": 3072,
    # The fraction of the rung ONE component may occupy when it is held WHOLE on a device, beside
    # the per-component overhead the estimator does not see. Read by `PrismSolver._usable_mb`,
    # which both `_place_component` (the whole-component placement under lazy_sequential and
    # friends) and `_try_layer_streaming` (what gets streamed, and the budget segments are cut
    # against) consult — one figure, so no component can fit neither.
    #
    # PROVENANCE: unrecorded. It entered as a literal `capacity_mb * 0.92` in `_place_component`
    # with v0.1.0-alpha (efe605b2) and was kept by 57b46db3; no measurement of 0.92 is in the
    # history. It is data here so the next measurement has one place to land, not a claim that
    # 0.92 is right.
    "whole_component_fraction": 0.92,
    # What the TRITON engine's arena holds live, over the profiled activation peak, while a
    # component runs. The profile measures the peak tensor set under the compiled engine's
    # caching allocator; the triton arena's live watermark runs above it — the tiling call site
    # in `PrismSolver._place_component` documents ~1.3x on CogVideoX-5b's VAE tile (2026-09:
    # 24 GB compiled, 31 GB+ triton for the same tile) and budgets TILES by it. The WHOLE test
    # did not, and mochi-1-preview's VAE at 85 frames 320x576 (profiled activation 24 561 MB,
    # 346 MB of weights) was placed whole on a 32 GB card and died at 25 202 MB live asking
    # 8 493 MB more — at least 33 695 MB, 1.35x the whole figure (2026-09-26, card 2). Read by
    # `PrismSolver._whole_component_mb` for the triton modes only; the compiled engine runs
    # under the allocator the profile measured.
    "triton_arena_activation_factor": 1.3,
    # The commercial memory ladder (Hocine's memory doctrine, 2026-09-21): the rungs a FREE
    # reading rounds DOWN onto on a shared pool, and the nominal rung of a dedicated card. Data,
    # 4 GB to 512 GB, read by `core/prism/memory_budget.py`; never a literal in the solver.
    "memory_ladder_gb": [4, 6, 8, 11, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 192, 256, 384, 512],
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
