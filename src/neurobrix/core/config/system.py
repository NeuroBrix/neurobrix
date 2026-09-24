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
def _ladder_gb(noise: float, lo_gb: int = 4, hi_gb: int = 512) -> list:
    """The memory ladder, spaced at the MEASURED noise of a free-memory reading.

    Consecutive rungs differ by `noise` (a fraction of the rung), so a swing of that size cannot
    carry a reading across more than one rung — which is what the rounding exists to guarantee —
    while discarding no more than the swing itself. A hand-picked absolute list cannot do both,
    because the noise is relative and the rungs are not.

    Values are GB and rounded to 3 decimals so the list stays readable; the low range is where
    the difference bites, and the top is unchanged in practice because the old rungs were already
    coarser than the noise there.
    """
    if not (0 < noise < 1):
        raise ValueError(f"memory_reading_noise={noise!r} is not a fraction between 0 and 1")
    out, x = [], float(lo_gb)
    while x < hi_gb:
        out.append(round(x, 3))
        x *= (1.0 + noise)
    # The top rung is the first geometric step AT or ABOVE `hi_gb`, never `hi_gb` pinned on as an
    # extra: pinning it created a final step of 1.76 % against a 2.55 % swing, which is a step a
    # reading can cross — the exact instability the rounding exists to prevent, reintroduced at
    # the one rung nobody would look at. The gate caught it on its first run.
    out.append(round(x, 3))
    return out


PRISM_DEFAULTS = {
    "safety_margin": 0.95,  # Use 95% of VRAM capacity (tight for large models)
    "default_seq_len": 128,  # Conservative default; actual value from defaults.json
    "overhead_factor": 0.05,  # 5% fragmentation buffer (low for inference)
    # Driver/library overhead reserve (MB) subtracted from a single GPU's
    # planning capacity before ANY strategy parks the whole model on one
    # device. Empirical derivation (P-PRISM-NEVER-REFUSE v2 B.4, 2026-05-12):
    # ~13 GiB of live NBX tensors produced a 16.6 GiB runtime peak on a
    # 32 GiB V100 — CUDA context + cuDNN/cuBLAS workspaces + Triton kernel
    # cache + caching-allocator fragmentation ≈ 3 GiB that no activation
    # estimator term covers. Single source for the reserve used by
    # _try_single_gpu, _try_single_gpu_lifecycle and _place_component.
    "oom_reserve_mb": 3072,
    # ── The memory ladder, and the measurement it is derived from ──────────────────────────
    #
    # The rungs a FREE reading rounds DOWN onto on a shared pool, and the nominal rung of a
    # dedicated card. Read by `core/prism/memory_budget.py`; never a literal in the solver.
    #
    # WHY IT ROUNDS AT ALL. A free reading is not a constant. Measured on this rack 2026-09-24,
    # host `MemAvailable`, 120 samples over 60 s with four cards busy — a LIVE machine, because
    # that is the condition the rounding defends against:
    #
    #     median      218 785 MB
    #     stdev         1 727 MB   (0.79 % of the median)
    #     p5..p95       5 366 MB   (2.45 %)
    #     full swing    5 571 MB   (2.55 %)
    #
    # A plan derived from an unrounded reading changes with the weather. This session watched
    # exactly that happen from the other side: the same Prism call returned
    # `single_gpu_lifecycle` during a 19-hour render holding ~19 GB of pinned host memory and
    # `single_gpu` afterwards, from byte-identical code (vacuous-gates register 99). So the
    # rounding stays, and only its SPACING is in question.
    #
    # WHY THE SPACING IS DERIVED AND NOT PICKED. The noise is RELATIVE — 2.55 % of the pool —
    # and hand-picked rungs are ABSOLUTE, so one list cannot be right at both ends. Against the
    # measured noise the old list's low range was 3.6x to 19.6x oversized:
    #
    #     rung MB   step MB   noise MB   step/noise
    #        4096      2048        104        19.6
    #        8192      3072        209        14.7      <- a reading of 10 638 fell to 8 192
    #       16384      4096        418         9.8      <- a reading of 17 277 fell to 16 384
    #
    # and the cost is memory discarded for nothing: 2 206 MB at the first, 822 MB at the second.
    # The second is not hypothetical — `granite-speech-3.3-8b`'s largest component is 16 769.6 MB
    # against a card reporting 17 277, and the rung is what put it 386 MB out of reach.
    #
    # So the ladder is GENERATED at the measured noise ratio: consecutive rungs differ by the
    # swing the reading actually shows, which is the smallest spacing that still absorbs it.
    # Re-measure `MEMORY_READING_NOISE` on new hardware and the ladder follows; do not edit rungs.
    "memory_reading_noise": 0.0255,   # full swing / median, measured; see above
    "memory_ladder_gb": _ladder_gb(0.0255, lo_gb=4, hi_gb=512),
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
