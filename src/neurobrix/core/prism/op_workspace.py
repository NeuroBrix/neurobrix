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


def op_workspace_fractions(memory: Dict[str, Any]) -> Dict[str, Any]:
    """{op type: fraction of its first input's bytes | None} from `memory.compiled_op_workspace`; the key missing, a
    non-table, a key that is not an ATen op name or a negative fraction is refused by name. None stands for "declared,
    not measured" (`op_workspace_fraction` refuses to price that op)."""
    if "compiled_op_workspace" not in memory:
        raise KeyError("ZERO FALLBACK: the vendor file states no memory.compiled_op_workspace — the table of what its "
                       "compiled ops hold beside input, output and operand copies while they run ({} when none does); "
                       "a compiled plan cannot be priced on it")
    table = memory["compiled_op_workspace"]
    if table is None:
        table = {}
    if not isinstance(table, dict):
        raise ValueError(f"ZERO FALLBACK: memory.compiled_op_workspace is {type(table).__name__}, not a table of op types")
    out: Dict[str, Any] = {}
    for op_type, f in table.items():
        if not (isinstance(op_type, str) and "::" in op_type):
            raise ValueError(f"ZERO FALLBACK: memory.compiled_op_workspace names {op_type!r}, which is not an op type (aten::...)")
        if f is None:
            out[op_type] = None
            continue
        f = float(f)
        if f < 0:
            raise ValueError(f"ZERO FALLBACK: memory.compiled_op_workspace[{op_type}] is {f}; a workspace is >= 0")
        out[op_type] = f
    return out


def op_workspace_fraction(table: Dict[str, Any], op_type: str, device_name: str = "this device") -> float:
    """The fraction for `op_type`, 0.0 for an op the table does not name; a `null` entry refuses by name."""
    f = table.get(op_type, 0.0)
    if f is None:
        raise RuntimeError(f"ZERO FALLBACK: {device_name}'s vendor file declares that its compiled {op_type} takes a workspace "
                           f"it has not measured (memory.compiled_op_workspace[{op_type}]: null) — a plan holding it cannot be "
                           f"priced there until the value is written")
    return float(f)
