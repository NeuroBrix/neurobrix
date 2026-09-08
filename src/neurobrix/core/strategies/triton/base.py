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


# Device-string prefixes that name an accelerator rather than host memory.
# Prism emits these from DeviceSpec.brand.to_device_prefix(): cuda for NVIDIA,
# hip for AMD, xpu for Intel, mps for Apple, tt for Tenstorrent. Matching only
# "cuda" here excluded every one of the others.
_ACCELERATOR_PREFIXES = frozenset({"cuda", "hip", "xpu", "mps", "tt", "musa", "npu"})


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
            prefix, _, idx = str(target_device).partition(":")
            if prefix == "cpu":
                return tensor.to_cpu()
            if prefix in _ACCELERATOR_PREFIXES:
                # `to_cuda` is the GENERIC accelerator move in this layer, not
                # an NVIDIA one: DeviceAllocator dispatches per backend and an
                # NBXTensor reports `_device == 'cuda'` on Metal as much as on
                # CUDA. Matching only the literal "cuda:" left every hip:/mps:
                # device falling through to `return tensor` — a silent no-op
                # that looks like a successful transfer.
                return tensor.to_cuda(int(idx) if idx.isdigit() else 0)
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
        """Synchronize `device` through DeviceAllocator (zero torch).

        Two things were wrong with the `startswith("cuda")` form, and only one
        of them was about non-NVIDIA hardware:

        * a `hip:1` / `mps:0` / `xpu:1` device did not match, so no
          `set_device` was issued and the call synchronised whatever card
          happened to be current — the wrong one, on a multi-GPU AMD box;
        * `set_device(idx)` was never undone. After
          `synchronize_device("cuda:1")` the current device was LEFT at 1, so
          the next operation that assumed the previous device ran on another
          card. That one bites multi-GPU NVIDIA exactly as hard, and it
          changes global state to do it.

        The device is restored on the way out, including if the sync raises.
        """
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        prefix, _, idx = str(device or "").partition(":")
        previous = None
        if prefix in _ACCELERATOR_PREFIXES:
            target = int(idx) if idx.isdigit() else 0
            try:
                previous = DeviceAllocator.get_device()
            except Exception:
                previous = None
            DeviceAllocator.set_device(target)
        try:
            DeviceAllocator.sync_device()
        finally:
            if previous is not None:
                DeviceAllocator.set_device(previous)
