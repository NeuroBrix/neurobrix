"""
NeuroBrix Prism - Hardware Supervisor
Multi-Interconnect Topology Support with Smart Allocation

Features:
- Best-Fit-Decreasing algorithm for heterogeneous GPUs
- Activation-aware memory estimation via ActivationProfiler
- Symbolic shape resolution without GPU allocation
"""

from .structure import (
    # Enums
    DeviceBrand,
    InterconnectTech,
    InterconnectType,
    AllocationStrategy,
    # Device/Topology
    DeviceSpec,
    InterconnectLink,
    InterconnectGroup,
    InterconnectTopology,
    PrismProfile,
    # Allocation structures
    PipelineStage,
    TensorParallelIntent,
    Zero3Config,
)
from .cpu_config import CPUConfig, apply_cpu_config
from .loader import load_profile, list_available_profiles
from .autodetect import load_default_profile, get_or_create_default_profile, detect_hardware
from .solver import (
    # Main solver
    PrismSolver,
    # Dataclasses — single source of truth
    ComponentAllocation,
    ComponentMemory,
    DeviceState,
    KVCachePlan,
    ExecutionPlan,
    # Backward compat
    PipelineExecutionPlan,
    # Import planning
    PrismImportPlanner,
    ImportPlan,
    # Convenience function
    solve,
)
from .profiler import (
    ActivationProfiler,
    ActivationProfile,
    InputConfig,
    profile_component,
)
from .memory_estimator import (
    compute_tensor_bytes,
    compute_dtype_factor,
    get_dtype_bytes_per_element,
    MemoryBreakdown,
)

# Backward compatibility aliases
GpuDevice = DeviceSpec
SmartSolver = PrismSolver
SmartExecutionPlan = PipelineExecutionPlan
smart_solve = solve
# Legacy alias: code importing StructComponentAllocation gets solver's version
StructComponentAllocation = ComponentAllocation
SolverExecutionPlan = ExecutionPlan

__all__ = [
    # Enums
    "DeviceBrand",
    "InterconnectTech",
    "InterconnectType",
    "AllocationStrategy",
    # Device/Topology
    "DeviceSpec",
    "GpuDevice",
    "InterconnectLink",
    "InterconnectGroup",
    "InterconnectTopology",
    "PrismProfile",
    # Allocation
    "PipelineStage",
    "TensorParallelIntent",
    "Zero3Config",
    "ComponentAllocation",
    "ExecutionPlan",
    # CPU Config
    "CPUConfig",
    "apply_cpu_config",
    # Loader
    "load_profile",
    "list_available_profiles",
    # Auto-detection
    "load_default_profile",
    "get_or_create_default_profile",
    "detect_hardware",
    # Solver
    "PrismSolver",
    "ComponentMemory",
    "DeviceState",
    "KVCachePlan",
    "PipelineExecutionPlan",
    "PrismImportPlanner",
    "ImportPlan",
    "solve",
    # Backward compatibility aliases
    "SmartSolver",
    "SmartExecutionPlan",
    "smart_solve",
    "StructComponentAllocation",
    "SolverExecutionPlan",
    # Activation Profiler
    "ActivationProfiler",
    "ActivationProfile",
    "InputConfig",
    "profile_component",
    # Memory Estimator
    "compute_tensor_bytes",
    "compute_dtype_factor",
    "get_dtype_bytes_per_element",
    "MemoryBreakdown",
]
