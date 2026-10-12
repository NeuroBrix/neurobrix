"""Triton execution strategies — NBXTensor-native (zero torch).

Full mirror of the PyTorch strategy registry for the triton execution path.
`get_strategy()` (parent package) routes here when `context.mode` is
"triton" / "triton_sequential". Every Prism strategy name has a triton-native
entry so any model that goes through the triton branch gets a zero-torch
placement strategy. (tp_sharding is NOT a strategy — it is the shard-creation
utility used by weight_sharding's "tp:" prefix.)

Two-modes doctrine: these are deliberate duplicates of core/strategies/*.py,
sharing only the StrategyContext contract — never compute code. The heavy
zero3 ratchet is reused via mixin (it is already NBXTensor-capable) rather
than duplicated.
"""

from .base import TritonStrategy
from .single_gpu import SingleGPUStrategy
from .component_placement import (
    ComponentPlacementStrategy,
    ComponentPlacementLazyStrategy,
)
from .pipeline_parallel import PipelineParallelStrategy
from .block_scatter import BlockScatterStrategy
from .weight_sharding import WeightShardingStrategy
from .lazy_sequential import LazySequentialStrategy
from .cpu_execution import CPUExecutionStrategy
from .zero3 import Zero3Strategy

# Triton-native strategies, keyed by the same Prism strategy names as the
# PyTorch registry. Complete: every registry name has a zero-torch entry.
TRITON_REGISTRY = {
    "single_gpu": SingleGPUStrategy,
    "single_gpu_lifecycle": SingleGPUStrategy,
    "component_placement": ComponentPlacementStrategy,
    "component_placement_lazy": ComponentPlacementLazyStrategy,
    "pipeline_parallel": PipelineParallelStrategy,
    "block_scatter": BlockScatterStrategy,
    "weight_sharding": WeightShardingStrategy,
    "lazy_sequential": LazySequentialStrategy,
    "zero3": Zero3Strategy,
    "cpu_execution": CPUExecutionStrategy,
}

__all__ = [
    "TritonStrategy",
    "SingleGPUStrategy",
    "ComponentPlacementStrategy",
    "ComponentPlacementLazyStrategy",
    "PipelineParallelStrategy",
    "BlockScatterStrategy",
    "WeightShardingStrategy",
    "LazySequentialStrategy",
    "CPUExecutionStrategy",
    "Zero3Strategy",
    "TRITON_REGISTRY",
]
