"""The workspace an op takes on a device BESIDE its input and output while it runs, as its vendor file states it.

`memory.compiled_conv3d_workspace_input_fraction`: the compiled engine's 3-D convolution, its workspace as a fraction of
its input's bytes. MPS's conv3d takes about its input's size on top (measured on an M4 Pro, 2026-10-17); cuDNN takes
what the allocator can give and no more (0). A vendor file that does not state it is refused by name, at the first
conv3d priced on it — no default. Torch-free: Prism reads it, the Triton branch may import it (R33)."""
from __future__ import annotations

from typing import Any, Dict


def conv3d_workspace_fraction(memory: Dict[str, Any]) -> float:
    if "compiled_conv3d_workspace_input_fraction" not in memory:
        raise KeyError("ZERO FALLBACK: the vendor file states no memory.compiled_conv3d_workspace_input_fraction — the "
                       "workspace its compiled conv3d takes beside its input and output; a plan holding a 3-D "
                       "convolution cannot be priced on it")
    f = float(memory["compiled_conv3d_workspace_input_fraction"])
    if f < 0:
        raise ValueError(f"ZERO FALLBACK: memory.compiled_conv3d_workspace_input_fraction is {f}; a workspace is >= 0")
    return f
