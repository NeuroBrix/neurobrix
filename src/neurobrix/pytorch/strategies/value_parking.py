"""A value between a streamed component's pieces, parked on the host and brought back — the runtime of
`Partition.swapped` (`LayerStreamingStrategy._run_pieces`, Prism's `LayerPartitioner._greedy_swapped`).

The PyTorch branch's (RED_LINES.md lines 16-22): a torch tensor moves with `.to(device)` (card to card through
`card_handoff`); anything that is not a tensor (a dict of inputs, a scalar) passes through unchanged; an NBXTensor is
refused by name (the Triton engine's are its twin's, `neurobrix.triton.strategies.value_parking`). A device name is
read by nbxtensor's `accelerator_index` (the library's one reading of a device name)."""
from __future__ import annotations

from typing import Any, Optional


def to_host(value: Any) -> Any:
    """`value` on the host (itself when it already is, or is no tensor)."""
    if hasattr(value, "to_cpu") and hasattr(value, "is_cuda"):          # NBXTensor
        raise RuntimeError("ZERO FALLBACK: value parking: the Triton branch's strategies are neurobrix.triton.strategies")
    # torch.Tensor on ANY accelerator: `is_cuda` is False on MPS, and a value "parked" by it stayed in Metal's pool —
    # mochi-1-preview's VAE at an 8 GB rung held three whole 2 981 MB maps on the device the plan had put on the host
    # (the Mac, 2026-10-27 12:06)
    dev = getattr(value, "device", None)
    kind = getattr(dev, "type", None) or str(dev or "cpu").split(":", 1)[0]
    if hasattr(value, "to") and dev is not None and kind != "cpu":
        return value.to("cpu")
    return value


def to_device(value: Any, device: Optional[str]) -> Any:
    """`value` on `device` (the piece's executor's), itself when it already is there or is no tensor."""
    from nbxtensor.backend.device_names import accelerator_index      # the one reading of a device name
    index = accelerator_index(device)
    if index is None:
        return value
    if hasattr(value, "to_cuda") and hasattr(value, "is_cuda"):          # NBXTensor
        raise RuntimeError("ZERO FALLBACK: value parking: the Triton branch's strategies are neurobrix.triton.strategies")
    if hasattr(value, "is_cuda") and hasattr(value, "to"):              # torch.Tensor
        if str(getattr(value, "device", "")) == str(device):
            return value
        from neurobrix.core.runtime.tensor_compat import is_torch_tensor
        if not is_torch_tensor(value):
            return value.to(device)
        from neurobrix.core.runtime.card_handoff import to_card         # card to card: never torch's own copy
        return to_card(value, device)
    return value
