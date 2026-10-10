"""The liveness of a graph — ONE rule for every engine that frees a tensor or prices its lifetime.

Three copies of this rule existed and drifted (stage B P1, 2026-10-10): the placement estimate
(core/prism/profiler.py) read an op's `input_tensor_ids`, the Triton sequence and the compiled
sequential executor read the tensors the op's `attributes` args/kwargs reference — the ones the
op actually receives. Over the 49 containers (610 811 ops) the two sources agree on every
activation; they differ only on 62 `aten::lift_fresh` ops of one graph, whose `param::` constant
named by `input_tensor_ids` is not the one the args pass — constants are never freed, so no
lifetime moves. The rule below reads the args/kwargs: what the runtime frees is what the
simulation prices.

A tensor dies after the last op that references it. A tensor produced and never referenced by any
op dies at the op that produced it (the dead-output rule: a text encoder's ~1 041 detach results,
a VAE's never-read conv-cache clones, a transformer's layer-norm statistics would otherwise live to
the end of the pass). The caller passes its own execution order — after its own fusions — and
removes what it never frees (weights, graph inputs, graph outputs) through `dead_at_op`.

Standard library only: imported by the Triton branch (R33) and by the compiled engine alike.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence


def _collect(arg: Any, out: List[str]) -> None:
    if not isinstance(arg, dict):
        return
    kind = arg.get("type")
    if kind in ("tensor", "tensor_ref"):
        tid = arg.get("tensor_id")
        if tid:
            out.append(tid)
    elif kind == "tensor_tuple":
        out.extend(arg.get("tensor_ids", []))
    elif kind == "list":
        for item in arg.get("value", []):
            _collect(item, out)


def op_tensor_refs(op: Mapping[str, Any]) -> List[str]:
    """The tensor ids an op receives, in argument order: its args, then its kwargs."""
    out: List[str] = []
    attrs = op.get("attributes") or {}
    for arg in attrs.get("args", []):
        _collect(arg, out)
    for arg in (attrs.get("kwargs") or {}).values():
        _collect(arg, out)
    return out


def last_uses(execution_order: Sequence[str], ops: Mapping[str, Mapping[str, Any]]) -> Dict[str, int]:
    """tensor id -> index in `execution_order` of the op after which it is dead."""
    last: Dict[str, int] = {}
    for idx, uid in enumerate(execution_order):
        op = ops.get(uid)
        if op is None:
            continue
        for tid in op_tensor_refs(op):
            last[tid] = idx
    for idx, uid in enumerate(execution_order):
        op = ops.get(uid)
        if op is None:
            continue
        for tid in op.get("output_tensor_ids", []):
            if tid not in last:
                last[tid] = idx
    return last


def slot_last_uses(last: Mapping[str, int], tid_to_slot: Mapping[str, int]) -> Dict[int, int]:
    """`last_uses` mapped onto an arena: a slot dies after the last op that needs ANY tensor it holds
    (a slot shared by aliases — an eliminated detach reuses its input's slot — lives to the latest)."""
    out: Dict[int, int] = {}
    for tid, idx in last.items():
        s = tid_to_slot.get(tid)
        if s is not None and idx > out.get(s, -1):
            out[s] = idx
    return out


def dead_at_op(last: Mapping[Any, int], protected: Iterable[Any]) -> Dict[int, List[Any]]:
    """op index -> the keys freed after it, `protected` (never freed) left out."""
    keep = set(protected)
    out: Dict[int, List[Any]] = {}
    for key, idx in last.items():
        if key not in keep:
            out.setdefault(idx, []).append(key)
    return out
