"""
NeuroBrix Prism - CPU Hardware Configuration

Data-driven CPU optimization from hardware profile YAML.
Reads cpu: section and produces thread/ISA/dtype settings.

ZERO HARDCODE: All values from cpu: section of hardware profile.
ZERO FALLBACK: Missing required cpu: fields = crash with explicit error.
"""

import os
import logging
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:  # R33: the ATen branch imports it; shared code only annotates
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
    # What the engine's device work adds to a process that has planned, before any weight — its modules,
    # the device context and the compute libraries a run's first ops load — MEASURED on this machine per
    # engine ({"compiled": MB, "triton": MB}, autodetect). Prism's host estimate adds it to the planning
    # process's own resident memory. Absent on a profile nobody measured: the estimate says so.
    runtime_base_mb: Dict[str, int] = field(default_factory=dict)
    # A hybrid CPU's two core classes (Apple silicon: performance + efficiency, `hw.perflevel0/1.physicalcpu`), when the
    # profile declares them; None on a CPU of one class.
    performance_cores: Optional[int] = None
    efficiency_cores: Optional[int] = None
    # Shard reads the weight loaders keep in flight, MEASURED on this machine (nbxtensor.io.read_rate); None: unmeasured
    io_workers: Optional[int] = None

    @property
    def ram_gb(self) -> float:
        return self.ram_mb / 1024

    @property
    def compute_threads(self) -> int:
        """The host's compute threads: the performance cores of a hybrid CPU (an efficiency core in a parallel region
        holds every other thread at its barrier), else the physical cores."""
        return int(self.performance_cores or self.cores)

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
            runtime_base_mb=dict(cpu_dict.get("runtime_base_mb") or {}),
            performance_cores=cpu_dict.get("performance_cores"),
            efficiency_cores=cpu_dict.get("efficiency_cores"),
            io_workers=cpu_dict.get("io_workers"),
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


def apply_thread_config(cpu: CPUConfig, involves_cpu: bool, mode: str = "compiled") -> None:
    """
    Set the host's thread configuration from the CPU profile — the ONE source (the owner's rule 1, 2026-10-26): every
    variable a compute library reads (OMP_NUM_THREADS, MKL_NUM_THREADS, TRITON_CPU_MAX_THREADS) and, on the ATen
    branch, torch's own pools. Applied whatever the plan: the host loads, converts and tokenizes on every plan, and a
    value left by anything else (an import-time `os.cpu_count()`, an inherited environment) is overwritten.

    Intra-op: `cpu.compute_threads` (the performance cores of a hybrid CPU, else the physical cores); inter-op:
    a quarter of them, at most 4. `involves_cpu` stays in the signature for the callers; it no longer gates.
    """
    intra_threads = cpu.compute_threads
    inter_threads = max(1, min(4, intra_threads // 4))
    import os
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TRITON_CPU_MAX_THREADS"):
        os.environ[var] = str(intra_threads)
    if mode in ("triton", "triton_sequential"):
        logger.info(f"[CPU] Threads: {intra_threads} from the profile for the Triton CPU backend — {cpu.model}")
        return
    import torch  # the ATen branch's thread pool
    torch.set_num_threads(intra_threads)
    try:
        torch.set_num_interop_threads(inter_threads)
    except RuntimeError:
        # torch refuses a second interop setting once its pool has run (a process that already computed)
        logger.debug(f"torch.set_num_interop_threads({inter_threads}) refused: the interop pool already ran")
    logger.info(f"[CPU] Threads: intra={intra_threads}, inter={inter_threads} from the profile — {cpu.model}")


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

    # ARM (Apple Silicon, Jetson) has native fp16/bf16 via NEON — skip x86 ISA checks
    if cpu.architecture in ("arm64", "aarch64"):
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
    mode: str = "compiled",
) -> None:
    """
    Apply all CPU optimizations from hardware profile.

    Single entry point — call after Prism solve, before execution; ``mode``
    names the engine whose host pool is configured.
    """
    involves_cpu = strategy_involves_cpu(strategy, device_count)
    apply_dnnl_isa(cpu)
    apply_thread_config(cpu, involves_cpu, mode)
    validate_dtype_isa(cpu, preferred_dtype)
