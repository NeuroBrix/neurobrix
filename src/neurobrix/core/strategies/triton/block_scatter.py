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

