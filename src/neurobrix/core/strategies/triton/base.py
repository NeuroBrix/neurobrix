"""Triton execution-strategy base — NBXTensor transfers, ZERO torch.

Mirror of the PyTorch `ExecutionStrategy` transfer surface; a device name is read and a tensor moved or a device
waited for by nbxtensor (`backend.device_names.to_named_device` / `synchronize_named`), R33: no `torch.*` on the
triton path. Strategy logic subclasses live alongside this file (`triton/single_gpu.py`, …) and inherit this base
instead of the torch one, so a triton-only NeuroBrix install carries no torch dependency on the placement layer.

The two-modes doctrine: `compiled` strategies (core/strategies/*.py) and
`triton` strategies (core/strategies/triton/*.py) are deliberately
duplicated, sharing only the `StrategyContext` contract — never compute
code. `get_strategy()` routes by `context.mode`.
"""

from typing import Dict, Any

from nbxtensor.backend.device_names import to_named_device, synchronize_named

from ..base import ExecutionStrategy


class TritonStrategy(ExecutionStrategy):
    """ExecutionStrategy whose tensor transfers + device sync are
    NBXTensor-native (no torch). Subclass for each placement strategy on
    the triton path."""

    def transfer_tensor(self, tensor: Any, target_device: str, async_transfer: bool = False) -> Any:
        """An NBXTensor (or a quantized one: what moves itself) on `target_device`; anything else (a python scalar carried
        in an input dict) unchanged."""
        return to_named_device(tensor, target_device) if hasattr(tensor, "to_cuda") and hasattr(tensor, "_device") \
            else tensor

    def transfer_dict(
        self,
        tensors: Dict[str, Any],
        target_device: str,
        async_transfer: bool = False,
    ) -> Dict[str, Any]:
        """Recursively move NBXTensor values in a dict to `target_device`.
        Same shape as the torch base, NBXTensor-typed (no torch isinstance)."""
        result: Dict[str, Any] = {}
        for key, value in tensors.items():
            if isinstance(value, dict):
                result[key] = self.transfer_dict(value, target_device, async_transfer)
            else:
                result[key] = self.transfer_tensor(value, target_device, async_transfer)
        return result

    def _release_device_cache(self, device: str) -> None:
        from nbxtensor.memory.allocator import DeviceAllocator
        DeviceAllocator.empty_cache_pool()

    # wait for `device` (the current one when it names no accelerator), the current device restored after
    synchronize_device = staticmethod(synchronize_named)
