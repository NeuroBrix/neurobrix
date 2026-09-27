"""The bytes each activation is STORED in — the DtypeEngine's own decision, walked over a graph.

Prism budgets a component's activations; the DtypeEngine decides, op by op, what dtype each
output is stored in. They were two answers: Prism priced every floating tensor at the compute
dtype while the engine's conservative path (a component with no calibration record, on fp16
hardware) stores matmul and AMP-fp32 outputs in fp32 and every elementwise op after them follows
its widest input. CogVideoX-5b-I2V's transformer, priced at 1 879 MB of activations, held ~4.1 GB
when its fp32 FFN hidden asked 1.20 GiB more on a 16 GB V100 (2026-09-27). This walk asks
`DtypeEngine.store_policy` — the function compile_op itself applies — so the plan and the run
cannot disagree about a dtype.
"""
from __future__ import annotations

from typing import Any, Dict

_FLOAT = ("float", "half", "bf16", "bfloat")
_FP32_POLICIES = frozenset({"fp32", "safe_softmax", "cpu_fp32"})
_COMPUTE_POLICIES = frozenset({"lower", "creation_fill"})


def _is_float(meta: Dict[str, Any]) -> bool:
    d = str((meta or {}).get("dtype", "")).lower()
    return any(t in d for t in _FLOAT) and "complex" not in d


def stored_bytes(graph: Dict[str, Any], engine, compute_bytes: int) -> Dict[str, int]:
    """{tensor_id: bytes per element} for every floating output the graph's ops write.

    `engine` is a DtypeEngine configured as the runtime configures it for this component
    (compute dtype and precision contract); `compute_bytes` is its compute dtype's width.
    Graph inputs, parameters and buffers are held in the compute dtype. A tensor the walk
    cannot decide (complex, a dtype copy) is absent: the caller keeps its own rule for it."""
    import torch
    tensors = graph.get("tensors", {}) or {}
    ops = graph.get("ops", {}) or {}
    width: Dict[str, int] = {}

    def of(tid: str) -> int:
        if tid in width:
            return width[tid]
        return compute_bytes if _is_float(tensors.get(tid, {})) else 0

    half_mul_guard = engine.compute_dtype == torch.float16
    for uid in graph.get("execution_order", []) or []:
        op = ops.get(uid) or {}
        policy = engine.store_policy(op.get("op_type", ""), op.get("attributes", {}) or {}, uid)
        base = policy[:-len("+store_compute")] if policy.endswith("+store_compute") else policy
        if policy.endswith("+store_compute") or base in _COMPUTE_POLICIES:
            w = compute_bytes
        elif base in _FP32_POLICIES or (base == "mul_safe" and half_mul_guard):
            w = 4
        elif base in ("promote", "passthrough", "mul_safe"):
            ins = [of(t) for t in op.get("input_tensor_ids", []) or []]
            ins = [b for b in ins if b]
            w = max(ins) if ins else compute_bytes
        else:
            continue                      # complex, to_copy: their own dtype rule
        for out in op.get("output_tensor_ids", []) or []:
            if _is_float(tensors.get(out, {})):
                width[out] = w
    return width
