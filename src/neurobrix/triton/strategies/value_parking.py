"""A value between a streamed component's pieces, parked on the host and brought back — the runtime of
`Partition.swapped` (`LayerStreamingStrategy._run_pieces`, Prism's `LayerPartitioner._greedy_swapped`).

The Triton branch's twin of `pytorch/strategies/value_parking.py`, Torch-free (RED_LINES.md lines 16-19): an
NBXTensor moves with `to_cpu(pinned=True)` / `to_cuda(index)` (the pinned host buffer is what an upload reads
fastest); anything that is not a tensor (a dict of inputs, a scalar) passes through unchanged; a tensor of another
class is refused by name. A device name is read by nbxtensor's `accelerator_index`."""
from __future__ import annotations

from typing import Any, Optional


def _refuse_foreign(value: Any) -> Any:
    """`value` itself when it is no tensor; a tensor that is no NBXTensor is refused by name."""
    if hasattr(value, "device") and hasattr(value, "to"):
        raise RuntimeError(f"ZERO FALLBACK: the Triton branch parks NBXTensors only, got {type(value).__name__}")
    return value


def to_host(value: Any) -> Any:
    """`value` on the host (itself when it already is, or is no tensor)."""
    if hasattr(value, "to_cpu") and hasattr(value, "is_cuda"):          # NBXTensor
        on_device = value.is_cuda or str(getattr(value, "_device", "cpu")) not in ("cpu", "")
        return value.to_cpu(pinned=True) if on_device else value
    return _refuse_foreign(value)


def to_device(value: Any, device: Optional[str]) -> Any:
    """`value` on `device` (the piece's executor's), itself when it already is there or is no tensor."""
    from nbxtensor.backend.device_names import accelerator_index      # the one reading of a device name
    index = accelerator_index(device)
    if index is None:
        return value
    if hasattr(value, "to_cuda") and hasattr(value, "is_cuda"):          # NBXTensor
        return value if value.is_cuda else value.to_cuda(index)
    return _refuse_foreign(value)
