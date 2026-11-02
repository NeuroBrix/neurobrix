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
from neurobrix.core.device_utils import device_empty_cache

from .base import ExecutionStrategy, StrategyContext
from neurobrix.core.strategies.shared_steps import BlockScatterSteps   # the steps both branches share


class BlockScatterStrategy(BlockScatterSteps, ExecutionStrategy):
    """
    Block scatter execution across multiple GPUs.

    Blocks within a component are distributed across GPUs using best-fit-decreasing.
    Unlike PipelineParallel (sequential fill), blocks may land on any GPU.
    Tensor transfers happen at block boundaries when crossing GPU boundaries.
    """


    def unload_weights(self, component_name: str) -> None:
        """Unload weights for scattered component."""
        executor = self.context.component_executors.get(component_name)
        if executor is not None:
            weights_loaded = hasattr(executor, '_weights') and bool(executor._weights)
            if weights_loaded:
                devices = self.get_component_devices(component_name)
                if hasattr(executor, '_is_fgp_mode'):
                    executor._is_fgp_mode = False
                executor.unload_weights()
                self._loaded_components.discard(component_name)
                for device in devices:
                    self.synchronize_device(device)
                device_empty_cache(device)


