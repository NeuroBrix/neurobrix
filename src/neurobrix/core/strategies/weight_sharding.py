"""
Weight Sharding Execution Strategy

Distributes weight files across multiple GPUs via shard_map (round-robin).
Execution uses CompiledSequence multi-device path with automatic
device alignment at op boundaries.

Use case: Component too large for single GPU but topology supports
multi-GPU execution. Weight files are distributed evenly across GPUs;
CompiledSequence handles cross-device transfers at op boundaries.
"""

from typing import Dict, Any, Optional, List, Set
import torch

from .base import ExecutionStrategy, StrategyContext
from neurobrix.core.strategies.shared_steps import WeightShardingSteps   # the steps both branches share


class WeightShardingStrategy(WeightShardingSteps, ExecutionStrategy):
    """
    Weight sharding execution across multiple GPUs.

    Weight files distributed via shard_map (from Prism solver, round-robin).
    Execution through CompiledSequence multi-device path.
    Device alignment handled by _run_inner_multi_device fast path.
    """


