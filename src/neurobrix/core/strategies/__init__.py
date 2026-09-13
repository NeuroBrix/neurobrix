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

import importlib

from .base import ExecutionStrategy, StrategyContext

# The compiled (ATen) strategy classes are imported when a name is resolved,
# never at package import: a --triton run takes its strategies from
# `.triton.TRITON_REGISTRY` and must not load the ATen branch (R33). The
# registry keeps the `name -> class` contract (membership, iteration,
# `registry[name] is Class`) through a lazy mapping.
_STRATEGY_CLASSES = {
    "SingleGPUStrategy": ".single_gpu",
    "ComponentPlacementStrategy": ".component_placement",
    "ComponentPlacementLazyStrategy": ".component_placement",
    "PipelineParallelStrategy": ".pipeline_parallel",
    "BlockScatterStrategy": ".block_scatter",
    "WeightShardingStrategy": ".weight_sharding",
    "LazySequentialStrategy": ".lazy_sequential",
    "LayerStreamingStrategy": ".layer_streaming",
    "Zero3Strategy": ".zero3",
    "CPUExecutionStrategy": ".cpu_execution",
}


#: Which strategies decide for themselves where their weights live.
#:
#: This is a DECLARATION, read without importing anything. Resolving the
#: class to read `manages_weight_residency` off it imports the ATen strategy
#: module — which a `--triton` run must never do (R33), and which this file's
#: own header says three lines up. Measured 2026-09-09: doing exactly that
#: broke every triton run with "Torch not compiled with CUDA enabled".
#:
#: The classes still declare the attribute, and
#: `test_the_table_and_the_classes_agree` fails if the two ever drift, so
#: this table cannot silently fall out of step with the strategies it names.
_MANAGES_WEIGHT_RESIDENCY = frozenset({
    "zero3",            # weights on pinned host memory, streamed for compute
    "layer_streaming",  # one segment of one component resident at a time
})


def strategy_manages_weight_residency(strategy_name: str) -> bool:
    """Does the named strategy decide for itself where its weights live?

    The runtime needs this for a component whose sub-strategy is named in the
    plan rather than being the plan's own strategy — `lazy_sequential` mapping
    one component to `zero3`, for instance. The NAME is used as a key into the
    registry, never as a test: what is read is the class's own declaration.

    Unknown names answer False rather than raising: the runtime asks this of
    every component, including ones whose allocation carries no sub-strategy
    at all, and an unknown name there means "nothing special", not an error.
    """
    return strategy_name in _MANAGES_WEIGHT_RESIDENCY


def _strategy_class(class_name: str):
    return getattr(importlib.import_module(_STRATEGY_CLASSES[class_name], __name__), class_name)


class _LazyRegistry(dict):
    """name -> strategy class; the class module is imported on first access."""

    def __getitem__(self, name):
        entry = dict.__getitem__(self, name)
        if isinstance(entry, str):
            entry = _strategy_class(entry)
            dict.__setitem__(self, name, entry)
        return entry

    def get(self, name, default=None):
        return self[name] if name in self else default

    def values(self):
        return [self[k] for k in self]

    def items(self):
        return [(k, self[k]) for k in self]


def __getattr__(name):
    if name in _STRATEGY_CLASSES:
        return _strategy_class(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# =============================================================================
# STRATEGY REGISTRY
# =============================================================================
# Every strategy name that Prism can emit MUST have a registry entry.
# ZERO FALLBACK: get_strategy() crashes on unknown names.
#
# Format: "strategy_name" → StrategyClass
# The solver uses these exact string names in ExecutionPlan.strategy.
# =============================================================================

STRATEGY_REGISTRY = _LazyRegistry({
    # === Single Device ===
    "single_gpu": "SingleGPUStrategy",
    "single_gpu_lifecycle": "SingleGPUStrategy",

    # === Component Placement (whole-component distribution) ===
    "component_placement": "ComponentPlacementStrategy",
    "component_placement_lazy": "ComponentPlacementLazyStrategy",

    # === Pipeline Parallel (per-layer sequential fill) ===
    "pipeline_parallel": "PipelineParallelStrategy",

    # === Block Scatter (block-level best-fit distribution) ===
    "block_scatter": "BlockScatterStrategy",

    # === Weight Sharding (weight-file round-robin) ===
    "weight_sharding": "WeightShardingStrategy",

    # === Sequential / Offload ===
    "lazy_sequential": "LazySequentialStrategy",
    "zero3": "Zero3Strategy",

    # === CPU-only (Doctrine R35 last-resort cascade) ===
    "cpu_execution": "CPUExecutionStrategy",
    # Same PLACEMENT as cpu_execution — every component on the host — and the
    # same class, exactly as single_gpu_lifecycle reuses SingleGPUStrategy.
    # What differs is the PLAN: the solver forces `loading_mode = "lazy"` so
    # the requirement drops from sum(components) to max(component), which is
    # the whole reason the rung exists. Two names, one placement, different
    # plan semantics.
    #
    # Added to the solver's cascade on 2026-09-03 and NOT registered here, so
    # the last rung of the ladder — the one that exists to guarantee a model
    # always runs — crashed with "Unknown strategy 'cpu_streaming'" the moment
    # it was selected. Caught by the CPU-only battery cell added in the same
    # session, on the full-zoo gate.
    "cpu_streaming": "CPUExecutionStrategy",

    # === Layer streaming (below every rung that keeps a component whole) ===
    # One segment of ONE component resident at a time. Reached only when
    # nothing above it is viable — it wins by scoring 50, never by a test on a
    # vendor, a device count or a memory size.
    #
    # It declares `manages_weight_residency`, which is how the runtime now
    # knows to drive it through `execute_component` and to offer it
    # `install_for_executor`. That question used to be "is this zero3", a
    # name, which is why this entry waited.
    "layer_streaming": "LayerStreamingStrategy",
})


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
    "strategy_manages_weight_residency",
    "ExecutionStrategy",
    "StrategyContext",
    "SingleGPUStrategy",
    "ComponentPlacementStrategy",
    "ComponentPlacementLazyStrategy",
    "PipelineParallelStrategy",
    "BlockScatterStrategy",
    "WeightShardingStrategy",
    "LazySequentialStrategy",
    "LayerStreamingStrategy",
    "Zero3Strategy",
    "CPUExecutionStrategy",
    "get_strategy",
    "STRATEGY_REGISTRY",
]
