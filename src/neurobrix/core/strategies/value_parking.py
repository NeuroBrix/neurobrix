"""A value between a streamed component's pieces, parked on the host and brought back — the runtime of
`Partition.swapped` (`LayerStreamingStrategy._run_pieces`, Prism's `LayerPartitioner._greedy_swapped`).

Both engines, by what the tensor offers: an NBXTensor moves with `to_cpu(pinned=True)` / `to_cuda(index)` (the
pinned host buffer is what an upload reads fastest), a torch tensor with `.to(device)`. No import of either (R33),
no conversion between them; anything that is not a tensor (a dict of inputs, a scalar) passes through unchanged."""
from __future__ import annotations

from typing import Any, Optional


def _index(device: Optional[str]) -> int:
    text = str(device or "")
    return int(text.split(":", 1)[1]) if ":" in text else 0


def to_host(value: Any) -> Any:
    """`value` on the host (itself when it already is, or is no tensor)."""
    if hasattr(value, "to_cpu") and hasattr(value, "is_cuda"):          # NBXTensor
        return value.to_cpu(pinned=True) if value.is_cuda else value
    if hasattr(value, "is_cuda") and hasattr(value, "to") and getattr(value, "is_cuda", False):   # torch.Tensor
        return value.to("cpu")
    return value


def to_device(value: Any, device: Optional[str]) -> Any:
    """`value` on `device` (the piece's executor's), itself when it already is there or is no tensor."""
    if device is None or not str(device).startswith(("cuda", "mps", "xpu", "hip")):
        return value
    if hasattr(value, "to_cuda") and hasattr(value, "is_cuda"):          # NBXTensor
        return value if value.is_cuda else value.to_cuda(_index(device))
    if hasattr(value, "is_cuda") and hasattr(value, "to"):              # torch.Tensor
        return value if str(getattr(value, "device", "")) == str(device) else value.to(device)
    return value
