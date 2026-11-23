"""Weight Sharding Execution Strategy — triton path (NBXTensor, zero torch).

Behavioural duplicate of pytorch/strategies/weight_sharding.py. Distributes
weight files across GPUs via the Prism shard_map (round-robin); the NBX
arena loads each shard to its device and the triton sequence aligns ops
at boundaries. Strategy-level logic is pure device routing — torch-free.
"""

from typing import Dict, Any, Optional, List, Set

from .base import StrategyContext
from .base import TritonStrategy
from .shared_steps import WeightShardingSteps   # the Triton twin of the PyTorch steps


class WeightShardingStrategy(WeightShardingSteps, TritonStrategy):
    """Weight sharding execution across multiple GPUs (triton path)."""


