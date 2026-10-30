"""Block Scatter Execution Strategy — triton path (NBXTensor, zero torch).

Behavioural duplicate of core/strategies/block_scatter.py. Blocks within a
component are distributed across GPUs (best-fit-decreasing) via the Prism
shard_map; the NBX weight loader places each block's weights on its device
and the triton sequence aligns ops at block boundaries. Source-device
detection uses NBXTensor `_device_idx` (no torch `.device`).
"""

from typing import Dict, Any, Optional, List, Set

from ..base import StrategyContext
from .base import TritonStrategy
from neurobrix.core.strategies.shared_steps import BlockScatterSteps   # the steps both branches share


def _nbx_device_str(tensor: Any) -> Optional[str]:
    """cuda:N string for an NBXTensor, else None. Zero torch."""
    if hasattr(tensor, "_device_idx") and hasattr(tensor, "_device"):
        dev = getattr(tensor, "_device", "cuda")
        if dev == "cpu":
            return "cpu"
        return f"cuda:{getattr(tensor, '_device_idx', 0)}"
    return None


class BlockScatterStrategy(BlockScatterSteps, TritonStrategy):
    """Block scatter execution across multiple GPUs (triton path)."""

    def __init__(self, context: StrategyContext, strategy_name: str = "block_scatter"):
        super().__init__(context, strategy_name)

        self.devices: List[str] = context.get_all_devices()
        self.use_async = True

        self.fgp_components: Dict[str, Dict[str, str]] = {}
        self.component_devices: Dict[str, List[str]] = {}

        for comp_name, alloc in context.allocations.items():
            if isinstance(alloc, dict):
                device_str = alloc.get('device', '')
                shard_map = alloc.get('shard_map', {})
            elif isinstance(alloc, tuple) and len(alloc) >= 2:
                device_str, shard_map = alloc[0], alloc[1]
            else:
                continue

            if isinstance(device_str, str) and device_str.startswith("fgp:"):
                device_list = device_str[4:].split(",")
                self.fgp_components[comp_name] = shard_map
                self.component_devices[comp_name] = device_list
            else:
                if isinstance(device_str, str):
                    self.component_devices[comp_name] = [device_str]

        if not self.devices:
            raise RuntimeError("ZERO FALLBACK: No devices assigned by Prism for block scatter")
        self.primary_device = self.devices[0]
        self._loaded_components: Set[str] = set()


    def unload_weights(self, component_name: str) -> None:
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
                from nbxtensor.memory.allocator import DeviceAllocator
                DeviceAllocator.empty_cache_pool()


