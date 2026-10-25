"""Whether an op is a vendor fp32 island the TRACE marks — ONE sentence for both engines' dtype policies
(`DtypeEngine.compile_op` and its sequential dispatch, `TritonDtypeEngine.wrap_op`).

In a half-precision graph, an op whose traced floating inputs are all float32 and whose traced floating output is
float32 ran in fp32 in the vendor's own forward: the vendor kept it there (a rotary table computed with autocast
disabled — Gemma's `inv_freq @ position_ids`, then cos and sin; a timestep frequency table). Its precision is the
trace's fact, not a range question: under bf16 compute the policy input-cast such a GEMM to bf16 ("bf16 range = fp32
range"), and the rotary angles of 506 positions lost ~2 rad — Sana_1600M_1024px_MultiLing's Gemma text encoder,
compiled branch on the M4 Pro, 2026-10-25, its image a different composition from the Triton branch's. Range stays
calibration's question (core/dtype/calibration.py: islands beyond finfo(dtype).max); this predicate only keeps what
the trace already computed in fp32 from being narrowed. Pure: no torch (R33).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

_HALF = ("float16", "bfloat16", "half")
_FLOATING = ("float16", "bfloat16", "float32", "float64", "half", "float", "double")
_FP32 = ("float32", "float")


def _name(d: Any) -> str:
    return str(d or "").replace("torch.", "")


def is_traced_fp32_island(op_record: Optional[Mapping[str, Any]], graph_dtype: Optional[str]) -> bool:
    """True when `op_record` (the op's graph entry: `input_dtypes`, `output_dtypes` at its top level) is an op of a
    half graph (`graph_dtype`, the container's traced torch_dtype) whose every floating input and floating output
    was traced float32 — at least one floating input. A cast out of the island (`_to_copy` fp32 -> half) is not
    one: its output is half."""
    if not op_record or _name(graph_dtype) not in _HALF:
        return False
    ins = [_name(d) for d in (op_record.get("input_dtypes") or ())]
    outs = [_name(d) for d in (op_record.get("output_dtypes") or ())]
    fin = [d for d in ins if d in _FLOATING]
    fout = [d for d in outs if d in _FLOATING]
    if not fin:
        return False
    return all(d in _FP32 for d in fin) and all(d in _FP32 for d in fout)
