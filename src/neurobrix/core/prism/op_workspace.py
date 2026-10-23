"""The workspace an op takes on a device BESIDE its input and output while it runs, as its vendor file states it.

`memory.compiled_conv3d_workspace_input_fraction`: the compiled engine's 3-D convolution, its workspace as a fraction of
its input's bytes. MPS's conv3d takes about its input's size on top (measured on an M4 Pro, 2026-10-17); cuDNN takes
what the allocator can give and no more (0). A vendor file that does not state it is refused by name, at the first
conv3d priced on it — no default. Torch-free: Prism reads it, the Triton branch may import it (R33)."""
from __future__ import annotations

from typing import Any, Dict


#: the dtype a convolution input of this many bytes per element runs in, where the width alone names it; a 2-byte
#: input runs in its component's half dtype (`conv3d_workspace_fraction(dtype=)`)
_WIDTH_DTYPE = {4: "float32", 8: "float64"}


def conv3d_dtype(x_width: int, compute_dtype: Any) -> str:
    """The dtype name a 3-D convolution reading `x_width`-byte elements runs in: fp32 / fp64 by width; a 2-byte input
    in its component's half compute dtype (bf16 and fp16 share the width and not the workspace: V100's compiled conv3d
    took 1.5-2 x its input in fp16 and 24 x in bf16, which it lowers to im2col, 2026-10-23)."""
    if int(x_width) in _WIDTH_DTYPE:
        return _WIDTH_DTYPE[int(x_width)]
    name = str(compute_dtype).replace("torch.", "").replace("NBXDtype.", "")
    if int(x_width) == 2 and name in ("float16", "bfloat16"):
        return name
    raise ValueError(f"ZERO FALLBACK: a 3-D convolution reads {x_width}-byte elements under a {name!r} compute dtype — "
                     f"no dtype names that width")


def conv3d_workspace_fraction(memory: Dict[str, Any], dtype: str = None) -> float:
    """The vendor file's fraction: one number for every dtype, or a table {dtype name: fraction} (float16, bfloat16,
    float32...) where the workspace depends on it — then the convolution's dtype is required and a dtype the table does
    not name is refused by name."""
    if "compiled_conv3d_workspace_input_fraction" not in memory:
        raise KeyError("ZERO FALLBACK: the vendor file states no memory.compiled_conv3d_workspace_input_fraction — the "
                       "workspace its compiled conv3d takes beside its input and output; a plan holding a 3-D "
                       "convolution cannot be priced on it")
    v = memory["compiled_conv3d_workspace_input_fraction"]
    if isinstance(v, dict):
        if dtype is None or dtype not in v:
            raise KeyError(f"ZERO FALLBACK: memory.compiled_conv3d_workspace_input_fraction is stated per dtype "
                           f"({sorted(v)}) and names no {dtype!r} — the workspace of a {dtype} conv3d is not measured")
        v = v[dtype]
    f = float(v)
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
