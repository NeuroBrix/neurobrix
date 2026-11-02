"""
Block Scatter Execution Strategy

Block-level best-fit distribution across multiple GPUs.
Individual transformer blocks distributed across GPUs using best-fit-decreasing.
Blocks can land on any GPU (not necessarily sequential).

Needed when a single component exceeds any single GPU's VRAM.

ARCHITECTURE:
- Prism provides shard_map: {"block.0.*": "cuda:0", "block.1.*": "cuda:1", ...}
- WeightLoader loads weights directly to their assigned GPUs
- GraphExecutor runs in FGP mode with device alignment per op
"""

from typing import Dict, Any, Optional, List, Set
import torch

from .base import ExecutionStrategy, StrategyContext
from neurobrix.core.strategies.shared_steps import BlockScatterSteps   # the steps both branches share


class BlockScatterStrategy(BlockScatterSteps, ExecutionStrategy):
    """
    Block scatter execution across multiple GPUs.

    Blocks within a component are distributed across GPUs using best-fit-decreasing.
    Unlike PipelineParallel (sequential fill), blocks may land on any GPU.
    Tensor transfers happen at block boundaries when crossing GPU boundaries.
    """

