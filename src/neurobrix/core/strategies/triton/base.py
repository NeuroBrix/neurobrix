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


def _cuda_index(target_device: str) -> int:
    """The card `target_device` names — `cuda` is card 0, `cuda:N` is card N.

    Anything else REFUSES here. The unrecognised string used to fall through and return the
    tensor unchanged: a transfer that moved nothing and read as success, which on a backend
    whose devices are not spelled `cuda` is every transfer of every request. The same silent
    pass-through already cost this surface once — `transfer_dict` matched only torch tensors,
    so on the triton path no NBXTensor was ever moved across devices.
    """
    if target_device == "cuda":
        return 0
    if target_device.startswith("cuda:"):
        return int(target_device.split(":", 1)[1])
    raise ValueError(
        f"NeuroBrix triton strategy: cannot transfer to {target_device!r} — this surface "
        f"moves NBXTensors between the engine's own devices (`cpu`, `cuda`, `cuda:N`). A "
        f"backend whose devices are spelled otherwise needs its case written here, not a "
        f"tensor handed back untouched.")


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
        if not (hasattr(tensor, "to_cuda") and hasattr(tensor, "_device")):
            return tensor                      # a scalar carried in an input dict
        if target_device == "cpu":
            return tensor.to_cpu()
        if target_device.startswith("zero3:"):
            # Residency on this rung belongs to the zero3 block loader, which moves a block
            # in and out around its own events. Named, so it is a decision and not the
            # fall-through it used to be.
            return tensor
        idx = _cuda_index(target_device)
        # The SHARED cross-device helper, not `to_cuda`: a D2D memcpy is enqueued on the
        # TARGET card's legacy stream and does not wait for the SOURCE card's stream, so a
        # peer copy issued straight after a component's kernels reads a buffer those kernels
        # may still be writing — wrong values, no error. `device_transfer.transfer_tensor`
        # syncs the source, enables peer access, materialises a non-dense window and carries
        # the strides; the op-by-op path has used it since the DeepSeek-Coder-V2-Lite and
        # Qwen3-Omni faults, and the component boundary went on calling `to_cuda` directly.
        from neurobrix.triton.device_transfer import transfer_tensor as _peer_transfer
        return _peer_transfer(tensor, idx)

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
        """Wait for the named card, and leave the current one where it was.

        `DeviceAllocator.sync_device()` syncs the CURRENT device and nothing else, so a
        caller that names a card must select it first — and put back what it found, because
        every allocation and launch that does not name a device lands on whatever this left
        current. With no device named there is nothing to select and the current card is the
        only honest answer; a caller on a multi-card plan should name one.

        Same discipline as `sequence._sync_deferred_all_devices`, which was written after the
        multi-GPU error-700 class.
        """
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        if device is None or not str(device).startswith("cuda"):
            DeviceAllocator.sync_device()
            return
        idx = _cuda_index(device)
        prev = DeviceAllocator.get_device()
        if prev == idx:
            DeviceAllocator.sync_device()
            return
        DeviceAllocator.set_device(idx)
        try:
            DeviceAllocator.sync_device()
        finally:
            DeviceAllocator.set_device(prev)
