"""Whether a component input enters at its TRACED width or at the compute dtype — ONE sentence for both engines
and for Prism (`TritonDtypeEngine._target_dtype_for_input`, `tensor_resolver.get_input_dtype`, `runtime_widths`).

An input traced wider than the graph's OWN parameters is a value the trace marks wider than the model — Allegro's
float32 timestep on bf16 weights (its sinusoidal embedding needs it whole; 777 became 776 in bf16, the halo of
2026-10-18) — and enters at its traced width, its first consumer casting it as that op's class dictates. An input
traced at the parameters' width is simply the model's width: a model stored and traced float32 that the policy runs
at fp16 (orpheus-3b, 2026-10-18: the lm_head's input entered float32 against fp16 weights and its first matmul
refused) enters at the compute dtype, as every input did before the rule. A graph without a floating parameter
(an embedding gather, a codec of buffers) is measured against the compute dtype itself. Pure: no torch (R33).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from neurobrix.core.dtype.itemsize import itemsize

_FLOATING_NAMES = ("float16", "bfloat16", "float32", "float64", "half", "float", "double")


def traced_parameter_width(graph_tensors: Mapping[str, Any]) -> Optional[int]:
    """Bytes per element of the widest floating parameter the graph was traced with, or None when it has none."""
    widest = None
    for tid, meta in (graph_tensors or {}).items():
        if not (tid.startswith("param::") or (meta or {}).get("is_parameter")):
            continue
        name = str((meta or {}).get("dtype") or "").replace("torch.", "")
        if name not in _FLOATING_NAMES:
            continue
        w = itemsize(name)
        widest = w if widest is None else max(widest, w)
    return widest


def enters_at_traced_width(input_itemsize: int, compute_itemsize: int, graph_tensors: Mapping[str, Any]) -> bool:
    """True when a floating input of `input_itemsize` bytes enters at its traced width: wider than the graph's
    parameters, or — a graph without a floating parameter — wider than the compute dtype."""
    reference = traced_parameter_width(graph_tensors)
    if reference is None:
        reference = compute_itemsize
    return int(input_itemsize) > int(reference)
