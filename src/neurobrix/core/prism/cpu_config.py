"""
NeuroBrix Prism - CPU Hardware Configuration

Data-driven CPU optimization from hardware profile YAML.
Reads cpu: section and produces thread/ISA/dtype settings.

ZERO HARDCODE: All values from cpu: section of hardware profile.
ZERO FALLBACK: Missing required cpu: fields = crash with explicit error.
"""

import os
import logging
import torch
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CPUConfig:
    """CPU hardware specification from hardware profile YAML."""
    model: str
    cores: int
    threads: int
    ram_mb: int
    architecture: str  # x86_64, aarch64, arm64
    features: List[str] = field(default_factory=list)

    @property
    def ram_gb(self) -> float:
        return self.ram_mb / 1024

    @classmethod
    def from_yaml_dict(cls, cpu_dict: dict) -> "CPUConfig":
        """Parse cpu: section from hardware profile YAML. ZERO FALLBACK."""
        required = ("model", "cores", "threads", "ram_mb", "architecture")
        for key in required:
            if key not in cpu_dict:
                raise RuntimeError(
                    f"ZERO FALLBACK: cpu.{key} missing in hardware profile.\n"
                    f"Regenerate profile: delete config/hardware/default.yml "
                    f"and run neurobrix without --hardware."
                )
        return cls(
            model=cpu_dict["model"],
            cores=cpu_dict["cores"],
            threads=cpu_dict["threads"],
            ram_mb=cpu_dict["ram_mb"],
            architecture=cpu_dict["architecture"],
            features=cpu_dict.get("features", []),
        )


# =============================================================================
# ISA → DNNL_MAX_CPU_ISA mapping
# Ordered from highest to lowest. First match wins.
# Source: oneDNN documentation — static hardware ISA hierarchy.
# =============================================================================
_ISA_HIERARCHY = [
    (["amx_bf16", "amx_int8"], "AVX512_CORE_AMX"),
    (["avx512_bf16"], "AVX512_CORE_BF16"),
    (["avx512_vnni"], "AVX512_CORE_VNNI"),
    (["avx512f"], "AVX512_CORE"),
    (["avx2"], "AVX2"),
    (["avx"], "AVX"),
    (["sse4_2"], "SSE41"),
]

# ISA features required for dtype correctness on CPU
_DTYPE_ISA_REQUIREMENTS = {
    "bfloat16": {"recommended": ["amx_bf16", "avx512_bf16"], "minimum": ["avx512f"]},
    "float16": {"recommended": ["f16c"], "minimum": ["avx2"]},
}

# =============================================================================
# CPU offload strategies that involve CPU in compute
# =============================================================================
_CPU_COMPUTE_STRATEGIES = frozenset({
    "zero3", "lazy_sequential", "component_placement_lazy",
})


def strategy_involves_cpu(strategy: str, device_count: int) -> bool:
    """Check if a strategy involves CPU in compute path."""
    if device_count == 0:
        return True  # CPU-only machine
    return strategy in _CPU_COMPUTE_STRATEGIES


def apply_thread_config(cpu: CPUConfig, involves_cpu: bool) -> None:
    """
    Set PyTorch thread configuration from CPU profile.

    Physical cores for intra-op (avoids hyperthreading contention).
    cores//4 (capped at 4) for inter-op parallelism.

    Only applies when CPU is involved in compute (offload or CPU-only).
    ZERO HARDCODE: cores from profile, not os.cpu_count().
    """
    if not involves_cpu:
        return

    intra_threads = cpu.cores
    inter_threads = max(1, min(4, cpu.cores // 4))

    try:
        torch.set_num_threads(intra_threads)
    except RuntimeError:
        # Already initialized — log but don't crash
        logger.debug(
            f"torch.set_num_threads({intra_threads}) failed — "
            f"PyTorch threading already initialized"
        )
        return

    try:
        torch.set_num_interop_threads(inter_threads)
    except RuntimeError:
        logger.debug(
            f"torch.set_num_interop_threads({inter_threads}) failed — "
            f"PyTorch interop threading already initialized"
        )

    logger.info(
        f"[CPU] Threads: intra={intra_threads} (physical cores), "
        f"inter={inter_threads} — {cpu.model}"
    )


def apply_dnnl_isa(cpu: CPUConfig) -> None:
    """
    Set DNNL_MAX_CPU_ISA based on detected CPU features.

    Never overrides user's DNNL_MAX_CPU_ISA if already set.
    ZERO HARDCODE: ISA derived from features list in profile.
    """
    if os.environ.get("DNNL_MAX_CPU_ISA"):
        logger.debug(f"DNNL_MAX_CPU_ISA already set: {os.environ['DNNL_MAX_CPU_ISA']}")
        return

    feature_set = set(cpu.features)
    for required_features, isa_name in _ISA_HIERARCHY:
        if any(f in feature_set for f in required_features):
            os.environ["DNNL_MAX_CPU_ISA"] = isa_name
            logger.info(f"[CPU] DNNL_MAX_CPU_ISA={isa_name}")
            return

    logger.debug(f"No recognized ISA features in {cpu.features}, DNNL_MAX_CPU_ISA not set")


def validate_dtype_isa(cpu: CPUConfig, preferred_dtype: Optional[str]) -> None:
    """
    Validate preferred_dtype compatibility with CPU ISA features.

    Warns if dtype chosen is suboptimal for the CPU.
    Does NOT crash — dtype may still work via software emulation.
    ZERO HARDCODE: requirements from _DTYPE_ISA_REQUIREMENTS.
    """
    if preferred_dtype is None:
        return

    requirements = _DTYPE_ISA_REQUIREMENTS.get(preferred_dtype)
    if requirements is None:
        return  # float32 always works

    feature_set = set(cpu.features)
    has_recommended = any(f in feature_set for f in requirements["recommended"])
    has_minimum = any(f in feature_set for f in requirements["minimum"])

    if not has_minimum:
        logger.warning(
            f"[CPU] preferred_dtype={preferred_dtype} but CPU lacks "
            f"minimum ISA features {requirements['minimum']}. "
            f"Performance may be severely degraded."
        )
    elif not has_recommended:
        logger.info(
            f"[CPU] preferred_dtype={preferred_dtype} — works but CPU lacks "
            f"optimal features {requirements['recommended']}"
        )


def should_pin_memory(cpu: CPUConfig, total_weight_mb: float) -> bool:
    """
    Decide whether to use pin_memory for CPU offload weights.

    Pinning doubles RAM usage temporarily. Skip if weights > 40% of RAM.
    ZERO HARDCODE: threshold derived from cpu.ram_mb.
    """
    if cpu.ram_mb <= 0:
        return False
    ram_budget_mb = cpu.ram_mb * 0.4
    if total_weight_mb > ram_budget_mb:
        logger.info(
            f"[CPU] Skipping pin_memory: weights {total_weight_mb:.0f}MB "
            f"> 40% of RAM ({ram_budget_mb:.0f}MB)"
        )
        return False
    return True


def apply_cpu_config(
    cpu: CPUConfig,
    strategy: str,
    device_count: int,
    preferred_dtype: Optional[str] = None,
) -> None:
    """
    Apply all CPU optimizations from hardware profile.

    Single entry point — call after Prism solve, before execution.
    """
    involves_cpu = strategy_involves_cpu(strategy, device_count)
    apply_dnnl_isa(cpu)
    apply_thread_config(cpu, involves_cpu)
    validate_dtype_isa(cpu, preferred_dtype)
