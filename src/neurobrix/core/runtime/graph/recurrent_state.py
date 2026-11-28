"""Pure-stdlib structural discovery of recurrent / causal-conv1d state ops.

Scans a normalized execution DAG for ``custom::gated_delta_rule`` ops and any
immediately upstream causal depthwise 1D convolution patterns feeding their
``q``/``k``/``v`` projections.

Zero torch / numpy / nbxtensor imports — safe to import from both the PyTorch
(``core/``) and Triton zero-Torch (``triton/``) branches.
Zero model-name hardcoding, zero magic layer counts or dimensions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

GDR_OP_TYPE = "custom::gated_delta_rule"
_CONV_OP_TYPES = frozenset({"aten::convolution", "aten::_convolution", "aten::conv1d"})

_UPSTREAM_WALK_OPS = frozenset({
    "aten::view", "aten::_unsafe_view", "aten::reshape", "aten::transpose",
    "aten::permute", "aten::contiguous", "aten::clone", "aten::_to_copy",
    "aten::detach", "aten::split", "aten::split_with_sizes", "aten::chunk",
    "aten::slice", "aten::narrow", "aten::select", "aten::unsqueeze",
    "aten::squeeze", "aten::expand", "aten::silu", "aten::gelu", "aten::relu",
    "aten::sigmoid", "aten::tanh", "aten::mul", "aten::add", "aten::rsqrt",
    "aten::sum", "aten::mean", "aten::pow", "custom::rms_norm",
})


@dataclass(frozen=True)
class GdrOpSpec:
    """Structural specification of one ``custom::gated_delta_rule`` op."""

    op_uid: str
    qk_l2norm: bool
    eps: float
    scale: Optional[float]


@dataclass(frozen=True)
class CausalConvSpec:
    """Structural specification of one causal depthwise 1D conv feeding GDR."""

    op_uid: str
    kernel_size: int
    channels: int
    mode: str
    pad_op_uid: Optional[str] = None


@dataclass(frozen=True)
class RecurrentGraphSpec:
    """All recurrent state ops discovered structurally in a component DAG."""

    gdr_ops: Tuple[GdrOpSpec, ...]
    conv_ops: Tuple[CausalConvSpec, ...]

    @property
    def is_empty(self) -> bool:
        return not self.gdr_ops and not self.conv_ops


def _arg_tensor_id(arg: Any) -> Optional[str]:
    return arg.get("tensor_id") if isinstance(arg, dict) and arg.get("type") == "tensor" else None


def _arg_scalar(arg: Any) -> Tuple[bool, Any]:
    if isinstance(arg, dict) and arg.get("type") == "scalar":
        return True, arg.get("value")
    if isinstance(arg, (int, float, bool)):
        return True, arg
    return False, None


def _arg_int_list(arg: Any) -> Optional[List[int]]:
    val = arg.get("value") if isinstance(arg, dict) and arg.get("type") == "list" else arg
    if isinstance(val, int) and not isinstance(val, bool):
        return [val]
    if isinstance(val, (list, tuple)) and all(isinstance(x, int) and not isinstance(x, bool) for x in val):
        return list(val)
    return None


def _op_args(op: Dict[str, Any]) -> List[Any]:
    return (op.get("attributes") or {}).get("args") or []


def _op_input_tids(op: Dict[str, Any]) -> List[str]:
    tids = op.get("input_tensor_ids")
    if tids:
        return list(tids)
    found: List[str] = []

    def _walk(x: Any) -> None:
        if isinstance(x, dict):
            if x.get("type") == "tensor" and "tensor_id" in x:
                found.append(x["tensor_id"])
            else:
                for v in x.values():
                    _walk(v)
        elif isinstance(x, list):
            for v in x:
                _walk(v)

    attrs = op.get("attributes") or {}
    _walk(attrs.get("args", []))
    _walk(attrs.get("kwargs", {}))
    return found


def _op_output_tids(op: Dict[str, Any], uid: str, tensors: Dict[str, Any]) -> List[str]:
    tids = op.get("output_tensor_ids")
    if tids:
        return list(tids)
    cand = f"{uid}::out_0"
    return [cand] if cand in tensors else []


def _tensor_shape(tid: str, tensors: Dict[str, Any]) -> Optional[List[int]]:
    meta = tensors.get(tid)
    shape = meta.get("shape") if isinstance(meta, dict) else None
    return list(shape) if isinstance(shape, (list, tuple)) else None


def _extract_conv_params(
    op: Dict[str, Any],
    tensors: Dict[str, Any],
) -> Optional[Tuple[str, int, int, int, int, int, bool]]:
    op_type = op.get("op_type", "")
    args = _op_args(op)
    attrs = op.get("attributes") or {}
    if len(args) < 2:
        return None
    x_tid, w_tid = _arg_tensor_id(args[0]), _arg_tensor_id(args[1])
    w_shape = _tensor_shape(w_tid, tensors) if (x_tid and w_tid) else None
    if not w_shape or len(w_shape) != 3:
        return None
    channels, in_per_group, kernel_size = int(w_shape[0]), int(w_shape[1]), int(w_shape[2])
    if channels <= 0 or in_per_group != 1 or kernel_size <= 1:
        return None

    if op_type == "aten::conv1d":
        stride_l = _arg_int_list(args[3]) if len(args) > 3 else _arg_int_list(attrs.get("stride", 1))
        pad_l = _arg_int_list(args[4]) if len(args) > 4 else _arg_int_list(attrs.get("padding", 0))
        dil_l = _arg_int_list(args[5]) if len(args) > 5 else _arg_int_list(attrs.get("dilation", 1))
        ok_g, groups = _arg_scalar(args[6]) if len(args) > 6 else (True, attrs.get("groups", 1))
        transposed = False
    else:
        stride_l = _arg_int_list(args[3]) if len(args) > 3 else _arg_int_list(attrs.get("stride", [1]))
        pad_l = _arg_int_list(args[4]) if len(args) > 4 else _arg_int_list(attrs.get("padding", [0]))
        dil_l = _arg_int_list(args[5]) if len(args) > 5 else _arg_int_list(attrs.get("dilation", [1]))
        ok_t, transposed = _arg_scalar(args[6]) if len(args) > 6 else (True, attrs.get("transposed", False))
        ok_g, groups = _arg_scalar(args[8]) if len(args) > 8 else (True, attrs.get("groups", 1))
        if not ok_t:
            return None

    if not ok_g or int(groups) != channels:
        return None
    if not (stride_l and pad_l and dil_l) or len(stride_l) != 1 or len(pad_l) != 1 or len(dil_l) != 1:
        return None
    return x_tid, channels, kernel_size, int(stride_l[0]), int(pad_l[0]), int(dil_l[0]), bool(transposed)


def _has_causal_tail_slice(
    conv_outs: List[str],
    x_tid: str,
    k_minus_1: int,
    ops: Dict[str, Any],
    tensors: Dict[str, Any],
    consumers_of: Dict[str, List[str]],
) -> bool:
    x_shape = _tensor_shape(x_tid, tensors)
    for otid in conv_outs:
        for cuid in consumers_of.get(otid, []):
            cop = ops.get(cuid)
            if not cop or cop.get("op_type") != "aten::slice":
                continue
            sargs = _op_args(cop)
            if len(sargs) < 4:
                continue
            ok_d, sdim = _arg_scalar(sargs[1])
            ok_s, sstart = _arg_scalar(sargs[2])
            if not (ok_d and ok_s and sdim in (2, -1) and sstart == 0):
                continue
            ok_e, send = _arg_scalar(sargs[3])
            if ok_e and send == -k_minus_1:
                return True
            couts = _op_output_tids(cop, cuid, tensors)
            c_shape = _tensor_shape(couts[0], tensors) if couts else None
            if x_shape and c_shape and len(x_shape) == 3 and len(c_shape) == 3 and x_shape[2] == c_shape[2]:
                return True
    return False


def _match_causal_conv(
    conv_uid: str,
    ops: Dict[str, Any],
    tensors: Dict[str, Any],
    producer_of: Dict[str, str],
    consumers_of: Dict[str, List[str]],
) -> Optional[CausalConvSpec]:
    op = ops.get(conv_uid)
    if not op or op.get("op_type") not in _CONV_OP_TYPES:
        return None
    params = _extract_conv_params(op, tensors)
    if params is None:
        return None
    x_tid, channels, kernel_size, stride, padding, dilation, transposed = params
    if transposed or stride != 1 or dilation != 1:
        return None
    k_minus_1 = kernel_size - 1

    if padding == k_minus_1 and _has_causal_tail_slice(
        _op_output_tids(op, conv_uid, tensors), x_tid, k_minus_1, ops, tensors, consumers_of
    ):
        return CausalConvSpec(op_uid=conv_uid, kernel_size=kernel_size, channels=channels, mode="pad_and_slice")

    if padding == 0:
        pad_uid = producer_of.get(x_tid)
        pad_op = ops.get(pad_uid) if pad_uid else None
        if pad_op and pad_op.get("op_type") == "aten::constant_pad_nd":
            pargs = _op_args(pad_op)
            if len(pargs) >= 2 and _arg_int_list(pargs[1]) == [k_minus_1, 0]:
                return CausalConvSpec(
                    op_uid=conv_uid, kernel_size=kernel_size, channels=channels,
                    mode="explicit_left_pad", pad_op_uid=pad_uid,
                )
    return None


def _build_graph_index(
    ops: Dict[str, Any], tensors: Dict[str, Any], order: List[str]
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    producer_of: Dict[str, str] = {}
    consumers_of: Dict[str, List[str]] = {}
    for uid in order:
        op = ops.get(uid)
        if not op:
            continue
        for tid in _op_output_tids(op, uid, tensors):
            producer_of[tid] = uid
        for tid in dict.fromkeys(_op_input_tids(op)):
            consumers_of.setdefault(tid, []).append(uid)
    return producer_of, consumers_of


def _find_upstream_causal_convs(
    seed_tids: List[str],
    ops: Dict[str, Any],
    tensors: Dict[str, Any],
    producer_of: Dict[str, str],
    consumers_of: Dict[str, List[str]],
    seen_convs: Set[str],
) -> List[CausalConvSpec]:
    found: List[CausalConvSpec] = []
    queue: List[Tuple[str, int]] = [(t, 0) for t in seed_tids]
    visited: Set[str] = set(seed_tids)
    while queue:
        tid, depth = queue.pop(0)
        puid = producer_of.get(tid) if depth <= 12 else None
        pop = ops.get(puid) if puid else None
        if not pop:
            continue
        ptype = pop.get("op_type", "")
        if ptype in _CONV_OP_TYPES:
            if puid not in seen_convs:
                cspec = _match_causal_conv(puid, ops, tensors, producer_of, consumers_of)
                if cspec is not None:
                    seen_convs.add(puid)
                    found.append(cspec)
            continue
        if ptype not in _UPSTREAM_WALK_OPS:
            continue
        for prev_tid in _op_input_tids(pop):
            if prev_tid not in visited:
                visited.add(prev_tid)
                queue.append((prev_tid, depth + 1))
    return found


def discover_recurrent_ops(dag: Dict[str, Any]) -> RecurrentGraphSpec:
    """Structurally discover all ``custom::gated_delta_rule`` and upstream causal conv1d ops."""
    ops: Dict[str, Any] = dag.get("ops") or {}
    tensors: Dict[str, Any] = dag.get("tensors") or {}
    order: List[str] = dag.get("execution_order") or []
    if not ops or not order:
        return RecurrentGraphSpec(gdr_ops=(), conv_ops=())

    producer_of, consumers_of = _build_graph_index(ops, tensors, order)
    gdr_specs: List[GdrOpSpec] = []
    conv_specs: List[CausalConvSpec] = []
    seen_convs: Set[str] = set()

    for uid in order:
        op = ops.get(uid)
        if not op or op.get("op_type") != GDR_OP_TYPE:
            continue
        attrs = op.get("attributes") or {}
        kw_attr = attrs.get("kwargs") or {}
        scale_raw = attrs.get("scale", kw_attr.get("scale", None))
        gdr_specs.append(GdrOpSpec(
            op_uid=uid,
            qk_l2norm=bool(attrs.get("qk_l2norm", kw_attr.get("qk_l2norm", True))),
            eps=float(attrs.get("eps", kw_attr.get("eps", 1e-6))),
            scale=float(scale_raw) if scale_raw is not None else None,
        ))
        conv_specs.extend(_find_upstream_causal_convs(
            _op_input_tids(op)[:3], ops, tensors, producer_of, consumers_of, seen_convs
        ))

    return RecurrentGraphSpec(gdr_ops=tuple(gdr_specs), conv_ops=tuple(conv_specs))
