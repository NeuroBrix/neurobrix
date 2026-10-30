"""The output dtype of an op (DESIGN rule 12: the mechanism in the library): read from the op's traced record, or
decided by the cast-back rule of an op computed in float32 inside a half forward. Pure functions of dtype names and of
a plain dict; the one definition every engine imports (an engine holding torch dtypes, NBXDtype or strings reads the
same answer: only the last dotted segment of a dtype's name is read).
"""
from __future__ import annotations

from typing import Any, Dict, FrozenSet, Optional

__all__ = ["COMPLEX_DTYPE_NAMES", "traced_output_dtype_name", "traced_output_is_complex", "amp_fp32_output_dtype"]

COMPLEX_DTYPE_NAMES: FrozenSet[str] = frozenset({"complex32", "complex64", "complex128", "chalf", "cfloat", "cdouble"})
_GRAPH_FLOAT_NAMES: FrozenSet[str] = frozenset({"float16", "bfloat16", "float32", "float64"})
_HALF_GRAPH_NAMES: FrozenSet[str] = frozenset({"float16", "bfloat16"})


def _name(dtype) -> str:
    return str(dtype).rsplit(".", 1)[-1]


def traced_output_dtype_name(record: Optional[Dict[str, Any]]) -> Optional[str]:
    """The plain name of the op's first traced output dtype (`output_dtypes[0]` at the record's TOP level), or None
    when the record states none."""
    if not isinstance(record, dict):
        return None
    dts = record.get("output_dtypes") or ()
    return _name(dts[0]) if dts else None


def traced_output_is_complex(record: Optional[Dict[str, Any]]) -> bool:
    """True when the op's traced record types an output complex.

    The traced OUTPUT dtype is the only signal: an op's name cannot answer (a plain `mul` by the scalar `1j` produces
    complex), nor can its inputs (that scalar is no input tensor). It matters because a half input at a
    complex-producing op yields complex32, which `exp`, `angle` and the inverse real FFT have no kernel for. Only
    `output_dtypes` at the record's TOP level is read: a nested `attributes` dict reads as "not complex"."""
    if not isinstance(record, dict):
        return False
    return any(_name(d) in COMPLEX_DTYPE_NAMES for d in (record.get("output_dtypes") or ()))


def amp_fp32_output_dtype(compute_dtype: str, graph_dtype: Optional[str], safe: bool,
                          narrowed: bool, traced: Optional[str] = None) -> str:
    """The OUTPUT dtype (a name) of an op computed in float32 inside under the half compute dtype `compute_dtype`.

    * bfloat16 on a bfloat16 graph (`graph_dtype`, the dtype the model was traced in): bfloat16 — a float32 result
      always fits bfloat16's range, and the model's own forward returns its input's dtype there.
    * bfloat16 on a graph of another floating dtype (a float32 model computed in bfloat16): float32, the dtype the
      model's forward returned. A graph dtype that is no floating dtype is refused by name.
    * float16: float16 only under the precision contract — `safe` (narrows every such op) or `narrowed` (this op is
      in the narrowed set); float32 otherwise. An op `traced` in float32 inside a half graph is the model's own
      float32 island: only `narrowed` brings it to float16, never `safe`."""
    if compute_dtype == "bfloat16":
        if graph_dtype not in _GRAPH_FLOAT_NAMES:
            raise ValueError(
                f"amp_fp32_output_dtype: under bfloat16 compute the output follows the graph dtype, and "
                f"{graph_dtype!r} is none of {sorted(_GRAPH_FLOAT_NAMES)} — the traced graph must state it")
        return "bfloat16" if graph_dtype == "bfloat16" else "float32"
    if compute_dtype == "float16":
        if traced == "float32" and graph_dtype in _HALF_GRAPH_NAMES:
            return "float16" if narrowed else "float32"
        return "float16" if (safe or narrowed) else "float32"
    raise ValueError(f"amp_fp32_output_dtype: compute dtype {compute_dtype!r} is not a half dtype — the cast-back "
                     f"rule exists for bfloat16 and float16 only")
