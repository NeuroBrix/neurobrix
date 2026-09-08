"""Triton execution-strategy base — NBXTensor transfers, ZERO torch.

Mirror of the PyTorch `ExecutionStrategy` transfer surface, but every
device move goes through `NBXTensor.to_cuda` / `to_cpu` and the
`DeviceAllocator` sync (R33: no `torch.*` on the triton path). Strategy
logic subclasses live alongside this file (`triton/single_gpu.py`, …) and
inherit this base instead of the torch one, so a triton-only NeuroBrix
install carries no torch dependency on the placement layer.

The two-modes doctrine: `compiled` strategies (core/strategies/*.py) and
`triton` strategies (core/strategies/triton/*.py) are deliberately
duplicated, sharing only the `StrategyContext` contract — never compute
code. `get_strategy()` routes by `context.mode`.
"""

from typing import Dict, Any, Optional

from ..base import ExecutionStrategy


class TritonStrategy(ExecutionStrategy):
    """ExecutionStrategy whose tensor transfers + device sync are
    NBXTensor-native (no torch). Subclass for each placement strategy on
    the triton path."""

    def transfer_tensor(
        self,
        tensor: Any,
        target_device: str,
        async_transfer: bool = False,
    ) -> Any:
        """Move an NBXTensor to `target_device` via DeviceAllocator. A
        non-NBXTensor (e.g. a python scalar carried in an input dict) is
        returned unchanged. No torch."""
        if hasattr(tensor, "to_cuda") and hasattr(tensor, "_device"):
            if target_device == "cpu":
                return tensor.to_cpu()
            if target_device.startswith("cuda:"):
                return tensor.to_cuda(int(target_device.split(":", 1)[1]))
            if target_device == "cuda":
                return tensor.to_cuda(0)
        return tensor

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
            if hasattr(value, "to_cuda") and hasattr(value, "_device"):
                result[key] = self.transfer_tensor(value, target_device, async_transfer)
            elif isinstance(value, dict):
                result[key] = self.transfer_dict(value, target_device, async_transfer)
            else:
                result[key] = value
        return result

    def synchronize_device(self, device: Optional[str] = None) -> None:
        """Synchronize a CUDA device through DeviceAllocator (zero torch)."""
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        if isinstance(device, str) and device.startswith("cuda"):
            idx = int(device.split(":", 1)[1]) if ":" in device else 0
            DeviceAllocator.set_device(idx)
        DeviceAllocator.sync_device()
