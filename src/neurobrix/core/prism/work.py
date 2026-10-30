"""The work one run of a component does at a request: its arithmetic and the bytes it moves (docs/design/
prism-time-score.md §2), read from its graph with the request's shapes — the same symbolic resolution the peak uses
(`ActivationProfiler`), never a run.

  flops            the multiply-adds of the ops that carry them, two flops each: products (mm, addmm, bmm, baddbmm,
                   matmul, linear), convolutions, attention (QK^T and PV) and the fused MoE's experts; every other op
                   counts one per output element
  weight_bytes     the parameters and buffers the run reads, once each, at their traced width
  activation_bytes every op's inputs read and outputs written that are not parameters, at their traced width; a view
                   (`_VIEWS`) moves nothing

The figures are the inputs of a roofline (`max(flops / throughput, bytes / bandwidth)`); the widths a component is
EXECUTED at (`core.prism.runtime_widths`) replace the traced ones where the caller has them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Dict, List, Mapping, Optional
from nbxtensor.shape import numel

_PRODUCTS = frozenset({"aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm", "aten::matmul", "aten::linear"})
_CONVS = frozenset({"aten::convolution", "aten::_convolution", "aten::conv1d", "aten::conv2d", "aten::conv3d"})
#: ops that move no bytes: a view of their input, read by whoever consumes it
_VIEWS = frozenset({"aten::view", "aten::_unsafe_view", "aten::reshape", "aten::permute", "aten::transpose", "aten::t",
                    "aten::expand", "aten::unsqueeze", "aten::squeeze", "aten::slice", "aten::select", "aten::as_strided",
                    "aten::alias", "aten::detach", "aten::flatten", "aten::unflatten", "aten::split",
                    "aten::split_with_sizes", "aten::chunk", "aten::narrow", "aten::unbind", "aten::view_as_real",
                    "aten::view_as_complex", "aten::lift_fresh"})
_ATTENTION = frozenset({"aten::scaled_dot_product_attention", "aten::_scaled_dot_product_attention",
                        "aten::_scaled_dot_product_efficient_attention", "aten::_scaled_dot_product_flash_attention",
                        "aten::_scaled_dot_product_flash_attention_for_cpu"})


@dataclass
class Work:
    flops: int = 0
    weight_bytes: int = 0
    activation_bytes: int = 0
    #: the bytes of the graph's output tensors at the request: what crosses a device boundary when the component's
    #: result (or, inside a transformer, its residual stream of the same width) is consumed on another device
    output_bytes: int = 0
    #: the routed experts one use of the component READS when its experts live off the card (an expert store):
    #: `top_k` experts per routed layer for a token, every expert the prompt's tokens touch for the prompt pass —
    #: set by the solver from the component's expert topology; 0 for a dense component or a resident one
    expert_bytes_per_use: int = 0

    def __add__(self, other: "Work") -> "Work":
        return Work(self.flops + other.flops, self.weight_bytes + other.weight_bytes,
                    self.activation_bytes + other.activation_bytes, self.output_bytes + other.output_bytes,
                    self.expert_bytes_per_use + other.expert_bytes_per_use)

    def times(self, n: int) -> "Work":
        return Work(self.flops * n, self.weight_bytes * n, self.activation_bytes * n, self.output_bytes * n,
                    self.expert_bytes_per_use * n)



def op_flops(kind: str, ins: List[List[int]], outs: List[List[int]], attrs: Mapping[str, Any]) -> int:
    """Two flops per multiply-add of the op, from its resolved operand and output shapes."""
    out = numel(outs[0]) if outs and outs[0] else 0
    if kind in _PRODUCTS and ins and ins[0]:
        # the contraction is the first operand's last dim (addmm / baddbmm: the second operand, the first is the bias)
        a = ins[1] if kind in ("aten::addmm", "aten::baddbmm") and len(ins) > 1 else ins[0]
        return 2 * out * int(a[-1]) if a else out
    if kind in _CONVS and len(ins) > 1 and ins[1]:
        return 2 * out * numel(ins[1][1:])                   # each output element: Cin/groups x kernel taps
    if kind in _ATTENTION and len(ins) >= 3 and ins[0] and ins[1]:
        q, k = ins[0], ins[1]
        lq, lk, d = int(q[-2]), int(k[-2]), int(q[-1])
        rows = numel(q[:-2])
        return 4 * rows * lq * lk * d                         # QK^T and PV
    if kind == "custom::moe_fused":
        h = ins[0] if ins else []
        tokens = numel(h[:-1]) if h else 0
        top_k = int(attrs.get("top_k") or 0)
        per_expert = int(attrs.get("_expert_params") or 0)
        return 2 * tokens * top_k * per_expert
    return out


def component_work(profiler, symbol_map: Dict[str, int], widths: Optional[Mapping[str, int]] = None,
                   ops: Optional[Collection[str]] = None) -> Work:
    """The work of one run of the profiler's graph at `symbol_map`. `widths`: {tensor id: bytes per element} as the
    engine executes it, where the caller has it. `ops`: only these op uids (a stretch of the graph), else every op."""
    tensors = profiler.tensors
    shapes: Dict[str, List[int]] = {}

    def shape(tid: str) -> List[int]:
        if tid not in shapes:
            shapes[tid] = profiler._resolve_shape(tensors[tid], symbol_map) if tid in tensors else []
        return shapes[tid]

    def size(tid: str) -> int:
        if tid not in tensors:
            return 0
        if widths and tid in widths:
            return numel(shape(tid)) * int(widths[tid])
        return profiler._compute_size(shape(tid), tensors[tid])

    def is_weight(tid: str) -> bool:
        meta = tensors.get(tid) or {}
        return bool(meta.get("is_parameter") or meta.get("is_buffer") or meta.get("constant"))

    w = Work()
    weights_seen = set()
    aliases = set()                                          # views of a weight: the weight, read once already
    for uid in profiler.execution_order:
        if ops is not None and uid not in ops:
            continue
        op = profiler.ops.get(uid) or {}
        ins = [t for t in (op.get("input_tensor_ids") or []) if t in tensors]
        outs = [t for t in (op.get("output_tensor_ids") or []) if t in tensors]
        attrs = dict(op.get("attributes") or {})
        if op.get("op_type") == "custom::moe_fused":
            ids = attrs.get("expert_gate_weight_ids") or []
            per = sum(numel(shape(t)) for key in ("expert_gate_weight_ids", "expert_up_weight_ids",
                                                   "expert_down_weight_ids") for t in (attrs.get(key) or [])[:1])
            attrs["_expert_params"] = per if ids else 0
        w.flops += op_flops(op.get("op_type", ""), [shape(t) for t in ins], [shape(t) for t in outs], attrs)
        view = op.get("op_type") in _VIEWS
        if view and any(is_weight(t) or t in aliases for t in ins):
            aliases.update(outs)
        for t in ins:
            if t in aliases:
                continue
            if is_weight(t):
                if t not in weights_seen:              # a weight is read once, whichever op (a view's transpose) meets it
                    weights_seen.add(t)
                    w.weight_bytes += size(t)
            elif not view:
                w.activation_bytes += size(t)
        if not view:
            for t in outs:
                w.activation_bytes += size(t)
    graph_outs = list((getattr(profiler, "dag", None) or {}).get("output_tensor_ids") or [])
    w.output_bytes = sum(size(t) for t in graph_outs if t in tensors)
    return w
