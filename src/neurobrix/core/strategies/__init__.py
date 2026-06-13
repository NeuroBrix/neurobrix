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
from .lazy_sequential import LazySequentialStrategy
from .zero3 import Zero3Strategy
from .cpu_execution import CPUExecutionStrategy


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

    # === CPU-only (Doctrine R35 last-resort cascade) ===
    "cpu_execution": CPUExecutionStrategy,
}


def get_strategy(strategy_name: str, context: StrategyContext) -> ExecutionStrategy:
    """
    Get strategy instance based on Prism's decision.

    Mode dispatch (two-modes doctrine): on the triton path
    (context.mode in {"triton", "triton_sequential"}) a NBXTensor-native
    strategy from `triton/TRITON_REGISTRY` is used when the strategy has
    been ported there; otherwise the PyTorch class is used (it already runs
    on the triton path via the polymorphic transfer helper). The compiled
    path always uses the PyTorch registry — byte-identical legacy behaviour.

    ZERO FALLBACK: Crash if the strategy name is unknown to BOTH registries.
    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise RuntimeError(
            f"ZERO FALLBACK: Unknown strategy '{strategy_name}'. "
            f"Available: {sorted(STRATEGY_REGISTRY.keys())}"
        )

    mode = getattr(context, "mode", "compiled")
    if mode in ("triton", "triton_sequential"):
        from .triton import TRITON_REGISTRY
        triton_class = TRITON_REGISTRY.get(strategy_name)
        if triton_class is not None:
            return triton_class(context, strategy_name)

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
    "CPUExecutionStrategy",
    "get_strategy",
    "STRATEGY_REGISTRY",
]
