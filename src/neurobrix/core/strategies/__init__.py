"""
NeuroBrix Execution Strategies

Strategy classes that handle component execution based on Prism allocation decisions.
The executor delegates to these classes — NO placement decisions in executor.

Strategy Hierarchy (granularity):
  Single GPU  →  Pipeline (component-level)  →  FGP (block-level)  →  TP (tensor-level)

Every strategy listed here is a first-class citizen — NeuroBrix is universal
and must handle any hardware combination. Prism scores ALL strategies and
selects the best viable one for the given hardware profile.

Strategies:
- SingleGPUStrategy: All components on one GPU
- PipelineStrategy: Components distributed across GPUs (PP)
- PipelineLazyStrategy: PP with lazy weight swap between phases
- FGPStrategy: Block-level distribution across GPUs (FGP)
- TensorParallelStrategy: Single component sharded across GPUs (TP)
- LazySequentialStrategy: One component at a time on largest GPU
- Zero3Strategy: CPU offload with GPU compute streaming
"""

from .base import ExecutionStrategy, StrategyContext
from .single_gpu import SingleGPUStrategy
from .pipeline import PipelineStrategy, PipelineLazyStrategy
from .fgp import FGPStrategy, FGPNVLinkStrategy, FGPPCIeStrategy
from .tensor_parallel import TensorParallelStrategy
from .zero3 import Zero3Strategy

# LazySequentialStrategy: load/unload components one at a time on the largest GPU.
# The solver allows GPU over-commitment because peak = max(single component),
# not sum(all components). Uses SingleGPU mechanics but with explicit unload.
LazySequentialStrategy = SingleGPUStrategy


# =============================================================================
# STRATEGY REGISTRY
# =============================================================================
# Every strategy name that Prism can emit MUST have a registry entry.
# ZERO FALLBACK: get_strategy() crashes on unknown names.
#
# Format: "strategy_name" → StrategyClass
# The solver uses these exact string names in ExecutionPlan.strategy.
# =============================================================================

STRATEGY_REGISTRY = {
    # === Single Device ===
    "single_gpu": SingleGPUStrategy,
    "single_gpu_lifecycle": SingleGPUStrategy,

    # === Pipeline Parallel (component-level distribution) ===
    "pp_nvlink": PipelineStrategy,
    "pp_pcie": PipelineStrategy,
    "pp_lazy_nvlink": PipelineLazyStrategy,
    "pp_lazy_pcie": PipelineLazyStrategy,

    # === Fine-Grained Pipeline (block-level distribution) ===
    "fgp_nvlink": FGPNVLinkStrategy,
    "fgp_pcie": FGPPCIeStrategy,

    # === Tensor Parallel (tensor-level sharding) ===
    "tp": TensorParallelStrategy,

    # === Sequential / Offload ===
    "lazy_sequential": LazySequentialStrategy,
    "zero3": Zero3Strategy,
}


def get_strategy(strategy_name: str, context: StrategyContext) -> ExecutionStrategy:
    """
    Get strategy instance based on Prism's decision.

    Args:
        strategy_name: Strategy name from Prism execution plan
        context: Shared context with allocations, executors, etc.

    Returns:
        ExecutionStrategy instance

    ZERO FALLBACK: Crash if unknown strategy
    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise RuntimeError(
            f"ZERO FALLBACK: Unknown strategy '{strategy_name}'. "
            f"Available: {sorted(STRATEGY_REGISTRY.keys())}"
        )

    strategy_class = STRATEGY_REGISTRY[strategy_name]
    return strategy_class(context, strategy_name)


__all__ = [
    "ExecutionStrategy",
    "StrategyContext",
    "SingleGPUStrategy",
    "PipelineStrategy",
    "PipelineLazyStrategy",
    "FGPStrategy",
    "FGPNVLinkStrategy",
    "FGPPCIeStrategy",
    "TensorParallelStrategy",
    "LazySequentialStrategy",
    "Zero3Strategy",
    "get_strategy",
    "STRATEGY_REGISTRY",
]
