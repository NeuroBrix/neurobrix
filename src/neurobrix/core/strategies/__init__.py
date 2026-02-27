"""
NeuroBrix Execution Strategies

Strategy classes that handle component execution based on Prism allocation decisions.
The executor delegates to these classes — NO placement decisions in executor.

Strategy Hierarchy (granularity):
  Single GPU → Component Placement → Pipeline Parallel (layer) → Block Scatter → Weight Sharding

Every strategy listed here is a first-class citizen — NeuroBrix is universal
and must handle any hardware combination. Prism scores ALL strategies and
selects the best viable one for the given hardware profile.

Strategies:
- SingleGPUStrategy: All components on one GPU
- ComponentPlacementStrategy: Whole components distributed across GPUs
- ComponentPlacementLazyStrategy: Component placement with lazy weight swap
- PipelineParallelStrategy: Per-layer sequential fill across GPUs (like Accelerate)
- BlockScatterStrategy: Block-level best-fit distribution across GPUs
- WeightShardingStrategy: Weight-file-level round-robin across GPUs
- LazySequentialStrategy: One component at a time on largest GPU
- Zero3Strategy: CPU offload with GPU compute streaming
"""

from .base import ExecutionStrategy, StrategyContext
from .single_gpu import SingleGPUStrategy
from .component_placement import ComponentPlacementStrategy, ComponentPlacementLazyStrategy
from .pipeline_parallel import PipelineParallelStrategy
from .block_scatter import BlockScatterStrategy
from .weight_sharding import WeightShardingStrategy
from .zero3 import Zero3Strategy

# LazySequentialStrategy: load/unload components one at a time on the largest GPU.
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

    # === Component Placement (whole-component distribution) ===
    "component_placement": ComponentPlacementStrategy,
    "component_placement_lazy": ComponentPlacementLazyStrategy,

    # === Pipeline Parallel (per-layer sequential fill) ===
    "pipeline_parallel": PipelineParallelStrategy,

    # === Block Scatter (block-level best-fit distribution) ===
    "block_scatter": BlockScatterStrategy,

    # === Weight Sharding (weight-file round-robin) ===
    "weight_sharding": WeightShardingStrategy,

    # === Sequential / Offload ===
    "lazy_sequential": LazySequentialStrategy,
    "zero3": Zero3Strategy,
}


def get_strategy(strategy_name: str, context: StrategyContext) -> ExecutionStrategy:
    """
    Get strategy instance based on Prism's decision.

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
    "ComponentPlacementStrategy",
    "ComponentPlacementLazyStrategy",
    "PipelineParallelStrategy",
    "BlockScatterStrategy",
    "WeightShardingStrategy",
    "LazySequentialStrategy",
    "Zero3Strategy",
    "get_strategy",
    "STRATEGY_REGISTRY",
]
