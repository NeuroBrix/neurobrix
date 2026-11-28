"""Generic graph fusion pass for unrolled ``chunk_gated_delta_rule`` subgraphs.

Replaces the fixed-chunk-count unrolled PyTorch decomposition of
``chunk_gated_delta_rule`` (triangular solve loop + N-chunk recurrence loop)
with a single dynamic ``custom::gated_delta_rule`` op that executes for ANY
sequence length ``L >= 1`` (prefill of arbitrary length AND decode ``L = 1``).

Pure Python standard library only — zero torch/numpy/engine imports, zero
model-specific constants (layer counts, head counts, head dims, chunk sizes,
scale, and eps are all read directly from the graph).
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

FUSED_OP = "custom::gated_delta_rule"

_VIEW_OPS = frozenset({"aten::view", "aten::_unsafe_view", "aten::reshape"})
_COPY_CLONE_OPS = frozenset({"aten::clone", "aten::_to_copy", "aten::contiguous", "aten::detach"})


# ---------------------------------------------------------------------------
# Graph inspection helpers (pure stdlib)
# ---------------------------------------------------------------------------

def _arg_tensor_id(arg: Any) -> Optional[str]:
    if isinstance(arg, dict) and arg.get("type") == "tensor":
        return arg.get("tensor_id")
    return None


def _arg_scalar(arg: Any) -> Tuple[bool, Any]:
    if isinstance(arg, dict) and arg.get("type") == "scalar":
        return True, arg.get("value")
    if isinstance(arg, (int, float, bool)):
        return True, arg
    return False, None


def _arg_list(arg: Any) -> Optional[List[Any]]:
    if isinstance(arg, dict) and arg.get("type") == "list":
        val = arg.get("value")
        return val if isinstance(val, list) else None
    if isinstance(arg, list):
        return arg
    return None


def _op_args(op: Dict[str, Any]) -> List[Any]:
    return op.get("attributes", {}).get("args", [])


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
    if not meta:
        return None
    shape = meta.get("shape")
    if isinstance(shape, list):
        return shape
    return None


def _tensor_dtype(tid: str, tensors: Dict[str, Any]) -> Optional[str]:
    meta = tensors.get(tid)
    return meta.get("dtype") if meta else None


def _build_indices(
    ops: Dict[str, Any],
    tensors: Dict[str, Any],
    order: List[str],
) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, int]]:
    producer_of: Dict[str, str] = {}
    consumers_of: Dict[str, List[str]] = {}
    order_idx: Dict[str, int] = {}
    for idx, uid in enumerate(order):
        order_idx[uid] = idx
        op = ops.get(uid)
        if not op:
            continue
        for tid in _op_output_tids(op, uid, tensors):
            producer_of[tid] = uid
        seen_in: Set[str] = set()
        for tid in _op_input_tids(op):
            if tid not in seen_in:
                seen_in.add(tid)
                consumers_of.setdefault(tid, []).append(uid)
    return producer_of, consumers_of, order_idx


def _strip_copies(tid: str, producer_of: Dict[str, str], ops: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Walk backwards through clone / _to_copy / contiguous / detach ops."""
    visited_ops: List[str] = []
    cur = tid
    while True:
        puid = producer_of.get(cur)
        if not puid:
            break
        pop = ops.get(puid)
        if not pop or pop.get("op_type") not in _COPY_CLONE_OPS:
            break
        args = _op_args(pop)
        if not args:
            break
        nxt = _arg_tensor_id(args[0])
        if not nxt:
            break
        visited_ops.append(puid)
        cur = nxt
    return cur, visited_ops


def _is_transpose_dims(op: Dict[str, Any], d0: int, d1: int) -> bool:
    if op.get("op_type") != "aten::transpose":
        return False
    args = _op_args(op)
    if len(args) < 3:
        return False
    ok1, a = _arg_scalar(args[1])
    ok2, b = _arg_scalar(args[2])
    if not (ok1 and ok2):
        return False
    return {a, b} == {d0, d1}


def _unwrap_pad_transpose(
    padded_tid: str,
    producer_of: Dict[str, str],
    ops: Dict[str, Any],
) -> Optional[Tuple[str, List[str]]]:
    """Walk ``constant_pad_nd -> [clone/_to_copy]* -> transpose(1, 2)`` back to ``[B, L, ...]``."""
    puid = producer_of.get(padded_tid)
    if not puid or ops[puid].get("op_type") != "aten::constant_pad_nd":
        return None
    pad_in = _arg_tensor_id(_op_args(ops[puid])[0]) if _op_args(ops[puid]) else None
    if not pad_in:
        return None
    cur, copy_ops = _strip_copies(pad_in, producer_of, ops)
    tuid = producer_of.get(cur)
    if not tuid or not _is_transpose_dims(ops[tuid], 1, 2):
        return None
    raw_tid = _arg_tensor_id(_op_args(ops[tuid])[0])
    if not raw_tid:
        return None
    return raw_tid, [puid] + copy_ops + [tuid]


def _unwrap_l2norm(
    tid: str,
    producer_of: Dict[str, str],
    ops: Dict[str, Any],
) -> Optional[Tuple[str, float, List[str]]]:
    """Match ``x * rsqrt(sum(x * x, dim=[-1], keepdim=True) + eps)``."""
    muid = producer_of.get(tid)
    if not muid or ops[muid].get("op_type") != "aten::mul":
        return None
    margs = _op_args(ops[muid])
    if len(margs) < 2:
        return None
    t0, t1 = _arg_tensor_id(margs[0]), _arg_tensor_id(margs[1])
    if not (t0 and t1):
        return None

    for x_tid, inv_tid in ((t0, t1), (t1, t0)):
        ruid = producer_of.get(inv_tid)
        if not ruid or ops[ruid].get("op_type") != "aten::rsqrt":
            continue
        r_in = _arg_tensor_id(_op_args(ops[ruid])[0]) if _op_args(ops[ruid]) else None
        if not r_in:
            continue
        auid = producer_of.get(r_in)
        if not auid or ops[auid].get("op_type") != "aten::add":
            continue
        aargs = _op_args(ops[auid])
        if len(aargs) < 2:
            continue
        sum_tid: Optional[str] = None
        eps_val: Optional[float] = None
        for idx_t, idx_s in ((0, 1), (1, 0)):
            ct = _arg_tensor_id(aargs[idx_t])
            ok, cs = _arg_scalar(aargs[idx_s])
            if ct and ok and isinstance(cs, (int, float)):
                sum_tid = ct
                eps_val = float(cs)
                break
        if sum_tid is None or eps_val is None:
            continue
        suid = producer_of.get(sum_tid)
        if not suid or ops[suid].get("op_type") != "aten::sum":
            continue
        sargs = _op_args(ops[suid])
        if len(sargs) < 2:
            continue
        sq_tid = _arg_tensor_id(sargs[0])
        dims = _arg_list(sargs[1])
        if not sq_tid or dims not in ([-1], [3]):
            continue
        squ = producer_of.get(sq_tid)
        if not squ or ops[squ].get("op_type") != "aten::mul":
            continue
        sqargs = _op_args(ops[squ])
        if len(sqargs) < 2:
            continue
        if _arg_tensor_id(sqargs[0]) == x_tid and _arg_tensor_id(sqargs[1]) == x_tid:
            return x_tid, eps_val, [muid, ruid, auid, suid, squ]
    return None


def _unwrap_head_expand(
    tid: str,
    producer_of: Dict[str, str],
    ops: Dict[str, Any],
    tensors: Dict[str, Any],
) -> Optional[Tuple[str, int, List[str]]]:
    """Match ``view([B, L, Hv, D]) <- [clone] <- expand([B, L, Hk, G, D]) <- unsqueeze(dim=3)``."""
    vuid = producer_of.get(tid)
    if not vuid or ops[vuid].get("op_type") not in _VIEW_OPS:
        return None
    v_in = _arg_tensor_id(_op_args(ops[vuid])[0]) if _op_args(ops[vuid]) else None
    if not v_in:
        return None
    cur, copy_ops = _strip_copies(v_in, producer_of, ops)
    euid = producer_of.get(cur)
    if not euid or ops[euid].get("op_type") != "aten::expand":
        return None
    e_in = _arg_tensor_id(_op_args(ops[euid])[0]) if _op_args(ops[euid]) else None
    if not e_in:
        return None
    uuid = producer_of.get(e_in)
    if not uuid or ops[uuid].get("op_type") != "aten::unsqueeze":
        return None
    uargs = _op_args(ops[uuid])
    if len(uargs) < 2:
        return None
    ok, udim = _arg_scalar(uargs[1])
    if not ok or udim not in (3, -2):
        return None
    orig_tid = _arg_tensor_id(uargs[0])
    if not orig_tid:
        return None
    orig_shape = _tensor_shape(orig_tid, tensors)
    out_shape = _tensor_shape(tid, tensors)
    if not (orig_shape and out_shape and len(orig_shape) == 4 and len(out_shape) == 4):
        return None
    hk, hv = int(orig_shape[2]), int(out_shape[2])
    if hk <= 0 or hv % hk != 0 or int(orig_shape[3]) != int(out_shape[3]):
        return None
    return orig_tid, hk, [vuid] + copy_ops + [euid, uuid]


def _unwrap_bmm_operand(
    tid: str,
    producer_of: Dict[str, str],
    ops: Dict[str, Any],
    expect_transpose: bool = False,
) -> Optional[str]:
    """Walk ``bmm`` operand backwards through ``view <- [expand] <- [transpose(-1, -2)]``."""
    cur = tid
    puid = producer_of.get(cur)
    if puid and ops[puid].get("op_type") in _VIEW_OPS:
        nxt = _arg_tensor_id(_op_args(ops[puid])[0])
        if nxt:
            cur = nxt
            puid = producer_of.get(cur)
    if puid and ops[puid].get("op_type") == "aten::expand":
        nxt = _arg_tensor_id(_op_args(ops[puid])[0])
        if nxt:
            cur = nxt
            puid = producer_of.get(cur)
    if expect_transpose:
        if not puid or not _is_transpose_dims(ops[puid], -1, -2):
            return None
        nxt = _arg_tensor_id(_op_args(ops[puid])[0])
        if not nxt:
            return None
        cur = nxt
    return cur


def _find_decay_mask(
    g_cumsum_tid: str,
    consumers_of: Dict[str, List[str]],
    ops: Dict[str, Any],
    tensors: Dict[str, Any],
) -> Optional[str]:
    """Match ``g_cumsum -> unsqueeze(-1)/(-2) -> sub -> tril -> exp -> tril``."""
    unsq_m1: Optional[str] = None
    unsq_m2: Optional[str] = None
    for cuid in consumers_of.get(g_cumsum_tid, []):
        op = ops[cuid]
        if op.get("op_type") != "aten::unsqueeze":
            continue
        args = _op_args(op)
        if len(args) < 2:
            continue
        ok, dim = _arg_scalar(args[1])
        if not ok:
            continue
        outs = _op_output_tids(op, cuid, tensors)
        if not outs:
            continue
        if dim in (-1, 4):
            unsq_m1 = outs[0]
        elif dim in (-2, 3):
            unsq_m2 = outs[0]
    if not (unsq_m1 and unsq_m2):
        return None

    sub_tid: Optional[str] = None
    for cuid in consumers_of.get(unsq_m1, []):
        op = ops[cuid]
        if op.get("op_type") == "aten::sub":
            args = _op_args(op)
            if len(args) >= 2 and _arg_tensor_id(args[0]) == unsq_m1 and _arg_tensor_id(args[1]) == unsq_m2:
                outs = _op_output_tids(op, cuid, tensors)
                if outs:
                    sub_tid = outs[0]
                    break
    if not sub_tid:
        return None

    cur = sub_tid
    for expected in ("aten::tril", "aten::exp", "aten::tril"):
        nxt = None
        for cuid in consumers_of.get(cur, []):
            op = ops[cuid]
            if op.get("op_type") == expected:
                outs = _op_output_tids(op, cuid, tensors)
                if outs:
                    nxt = outs[0]
                    break
        if not nxt:
            return None
        cur = nxt
    return cur


def _next_select_scatter_output(
    cur: str,
    consumers_of: Dict[str, List[str]],
    ops: Dict[str, Any],
    tensors: Dict[str, Any],
    dim: int,
) -> Optional[str]:
    """Return the first ``aten::select_scatter(dim=dim)`` output consuming ``cur`` as arg 0."""
    for cuid in consumers_of.get(cur, []):
        op = ops[cuid]
        if op.get("op_type") != "aten::select_scatter":
            continue
        args = _op_args(op)
        if len(args) < 4 or _arg_tensor_id(args[0]) != cur:
            continue
        ok, d = _arg_scalar(args[2])
        if ok and d == dim:
            outs = _op_output_tids(op, cuid, tensors)
            if outs:
                return outs[0]
    return None


def _find_select_scatter_tail(
    start_tid: str,
    consumers_of: Dict[str, List[str]],
    ops: Dict[str, Any],
    tensors: Dict[str, Any],
    dim: int = 2,
) -> Optional[str]:
    """Follow a chain of ``aten::select_scatter(dim=dim)`` from ``start_tid`` to its final output."""
    cur = start_tid
    seen: Set[str] = set()
    while cur not in seen:
        seen.add(cur)
        nxt = _next_select_scatter_output(cur, consumers_of, ops, tensors, dim)
        if nxt is None:
            return cur
        cur = nxt
    return None


def _try_extract_k_beta_from_mul(
    mul_op: Dict[str, Any],
    decay_mask_tid: str,
    producer_of: Dict[str, str],
    ops: Dict[str, Any],
) -> Optional[Tuple[str, str]]:
    """Extract ``(k_padded_tid, beta_padded_tid)`` from a ``mul(decay_mask, view(bmm(k_beta, k^T)))`` op."""
    margs = _op_args(mul_op)
    if len(margs) < 2:
        return None
    t0, t1 = _arg_tensor_id(margs[0]), _arg_tensor_id(margs[1])
    other = t1 if t0 == decay_mask_tid else (t0 if t1 == decay_mask_tid else None)
    ouid = producer_of.get(other) if other else None
    if not ouid or ops[ouid].get("op_type") not in _VIEW_OPS:
        return None
    bmm_tid = _arg_tensor_id(_op_args(ops[ouid])[0])
    bmm_uid = producer_of.get(bmm_tid) if bmm_tid else None
    if not bmm_uid or ops[bmm_uid].get("op_type") != "aten::bmm":
        return None
    bargs = _op_args(ops[bmm_uid])
    lhs_tid = _arg_tensor_id(bargs[0]) if len(bargs) >= 2 else None
    rhs_tid = _arg_tensor_id(bargs[1]) if len(bargs) >= 2 else None
    if not (lhs_tid and rhs_tid):
        return None
    k_beta_chunked = _unwrap_bmm_operand(lhs_tid, producer_of, ops, expect_transpose=False)
    k_chunked = _unwrap_bmm_operand(rhs_tid, producer_of, ops, expect_transpose=True)
    k_vuid = producer_of.get(k_chunked) if k_chunked else None
    kb_vuid = producer_of.get(k_beta_chunked) if k_beta_chunked else None
    if not (k_vuid and kb_vuid):
        return None
    if ops[k_vuid].get("op_type") not in _VIEW_OPS or ops[kb_vuid].get("op_type") not in _VIEW_OPS:
        return None
    cand_k_pad = _arg_tensor_id(_op_args(ops[k_vuid])[0])
    kb_mul_tid = _arg_tensor_id(_op_args(ops[kb_vuid])[0])
    kb_mul_uid = producer_of.get(kb_mul_tid) if kb_mul_tid else None
    if not (cand_k_pad and kb_mul_uid and ops[kb_mul_uid].get("op_type") == "aten::mul"):
        return None
    kbm_args = _op_args(ops[kb_mul_uid])
    if len(kbm_args) < 2:
        return None
    kt0, kt1 = _arg_tensor_id(kbm_args[0]), _arg_tensor_id(kbm_args[1])
    beta_unsq = kt1 if kt0 == cand_k_pad else (kt0 if kt1 == cand_k_pad else None)
    bu_uid = producer_of.get(beta_unsq) if beta_unsq else None
    if not bu_uid or ops[bu_uid].get("op_type") != "aten::unsqueeze":
        return None
    cand_beta_pad = _arg_tensor_id(_op_args(ops[bu_uid])[0])
    return (cand_k_pad, cand_beta_pad) if cand_beta_pad else None


def _find_k_beta_padded(
    decay_mask_tid: str,
    consumers_of: Dict[str, List[str]],
    producer_of: Dict[str, str],
    ops: Dict[str, Any],
) -> Optional[Tuple[str, str]]:
    """Find ``(k_padded_tid, beta_padded_tid)`` from ``decay_mask * (k_beta @ k^T)``."""
    for cuid in consumers_of.get(decay_mask_tid, []):
        op = ops[cuid]
        if op.get("op_type") != "aten::mul":
            continue
        found = _try_extract_k_beta_from_mul(op, decay_mask_tid, producer_of, ops)
        if found is not None:
            return found
    return None


def _find_v_padded(
    beta_padded_tid: str,
    k_padded_tid: str,
    consumers_of: Dict[str, List[str]],
    ops: Dict[str, Any],
    tensors: Dict[str, Any],
) -> Optional[str]:
    """Find ``v_padded_tid`` via the other ``unsqueeze(-1) -> mul`` consumer of ``beta_padded``."""
    for cuid in consumers_of.get(beta_padded_tid, []):
        op = ops[cuid]
        if op.get("op_type") != "aten::unsqueeze":
            continue
        uouts = _op_output_tids(op, cuid, tensors)
        if not uouts:
            continue
        bu_tid = uouts[0]
        for mcuid in consumers_of.get(bu_tid, []):
            mop = ops[mcuid]
            if mop.get("op_type") != "aten::mul":
                continue
            margs = _op_args(mop)
            if len(margs) < 2:
                continue
            mt0, mt1 = _arg_tensor_id(margs[0]), _arg_tensor_id(margs[1])
            other = mt1 if mt0 == bu_tid else (mt0 if mt1 == bu_tid else None)
            if other and other != k_padded_tid:
                return other
    return None


def _unwrap_optional_q_scale(
    q_scaled_tid: str, producer_of: Dict[str, str], ops: Dict[str, Any],
) -> Tuple[str, Optional[float]]:
    """Strip an optional scalar ``aten::mul(q_padded, scale)`` producing ``q_scaled_tid``."""
    qs_uid = producer_of.get(q_scaled_tid)
    if qs_uid and ops[qs_uid].get("op_type") == "aten::mul":
        qsargs = _op_args(ops[qs_uid])
        if len(qsargs) >= 2:
            for it, isc in ((0, 1), (1, 0)):
                ct = _arg_tensor_id(qsargs[it])
                ok, cs = _arg_scalar(qsargs[isc])
                if ct and ok and isinstance(cs, (int, float)):
                    return ct, float(cs)
    return q_scaled_tid, None


def _try_extract_q_from_dm0_mul(
    mop: Dict[str, Any],
    mcuid: str,
    dm0_tid: str,
    producer_of: Dict[str, str],
    ops: Dict[str, Any],
    tensors: Dict[str, Any],
) -> Optional[Tuple[str, Optional[float], Optional[str]]]:
    """Extract ``(q_padded_tid, scale_val, dm0_mul_tid)`` from a ``mul(dm0, view(bmm(q0, k0^T)))`` op."""
    margs = _op_args(mop)
    if len(margs) < 2:
        return None
    mt0, mt1 = _arg_tensor_id(margs[0]), _arg_tensor_id(margs[1])
    qk_tid = mt1 if mt0 == dm0_tid else (mt0 if mt1 == dm0_tid else None)
    qku = producer_of.get(qk_tid) if qk_tid else None
    if not qku or ops[qku].get("op_type") not in _VIEW_OPS:
        return None
    bmm_tid = _arg_tensor_id(_op_args(ops[qku])[0])
    bmm_uid = producer_of.get(bmm_tid) if bmm_tid else None
    if not bmm_uid or ops[bmm_uid].get("op_type") != "aten::bmm":
        return None
    lhs_tid = _arg_tensor_id(_op_args(ops[bmm_uid])[0])
    q0_tid = _unwrap_bmm_operand(lhs_tid, producer_of, ops, expect_transpose=False) if lhs_tid else None
    q0_uid = producer_of.get(q0_tid) if q0_tid else None
    if not q0_uid or ops[q0_uid].get("op_type") != "aten::select":
        return None
    q_chunked_tid = _arg_tensor_id(_op_args(ops[q0_uid])[0])
    q_vuid = producer_of.get(q_chunked_tid) if q_chunked_tid else None
    if not q_vuid or ops[q_vuid].get("op_type") not in _VIEW_OPS:
        return None
    q_scaled_tid = _arg_tensor_id(_op_args(ops[q_vuid])[0])
    if not q_scaled_tid:
        return None
    q_padded_tid, scale_val = _unwrap_optional_q_scale(q_scaled_tid, producer_of, ops)
    mouts = _op_output_tids(mop, mcuid, tensors)
    return q_padded_tid, scale_val, (mouts[0] if mouts else None)


def _find_q_padded_and_scale(
    decay_mask_tid: str,
    consumers_of: Dict[str, List[str]],
    producer_of: Dict[str, str],
    ops: Dict[str, Any],
    tensors: Dict[str, Any],
) -> Optional[Tuple[str, Optional[float], Optional[str]]]:
    """Find ``(q_padded_tid, scale_val, dm0_mul_tid)`` from ``decay_mask_0 * (q_0 @ k_0^T)``."""
    for cuid in consumers_of.get(decay_mask_tid, []):
        op = ops[cuid]
        if op.get("op_type") != "aten::select":
            continue
        sargs = _op_args(op)
        if len(sargs) < 3:
            continue
        ok1, sdim = _arg_scalar(sargs[1])
        ok2, sidx = _arg_scalar(sargs[2])
        if not (ok1 and ok2 and sdim == 2 and sidx == 0):
            continue
        dm0_outs = _op_output_tids(op, cuid, tensors)
        if not dm0_outs:
            continue
        dm0_tid = dm0_outs[0]
        for mcuid in consumers_of.get(dm0_tid, []):
            mop = ops[mcuid]
            if mop.get("op_type") != "aten::mul":
                continue
            found = _try_extract_q_from_dm0_mul(mop, mcuid, dm0_tid, producer_of, ops, tensors)
            if found is not None:
                return found
    return None


def _is_chunk0_select_scatter(cop: Dict[str, Any]) -> bool:
    """Return True if ``cop`` is ``aten::select_scatter(..., dim=2, index=0)``."""
    if cop.get("op_type") != "aten::select_scatter":
        return False
    sargs = _op_args(cop)
    if len(sargs) < 4:
        return False
    ok_d, sdim = _arg_scalar(sargs[2])
    ok_i, sidx = _arg_scalar(sargs[3])
    return bool(ok_d and ok_i and sdim == 2 and sidx == 0)


def _expand_bfs_frontier(
    bfs_q: List[str],
    bfs_seen: Set[str],
    consumers_of: Dict[str, List[str]],
    ops: Dict[str, Any],
    tensors: Dict[str, Any],
) -> Tuple[Optional[str], List[str]]:
    """Advance one BFS hop looking for ``select_scatter(dim=2, index=0)``."""
    nxt_q: List[str] = []
    for tid_c in bfs_q:
        for cuid in consumers_of.get(tid_c, []):
            cop = ops[cuid]
            outs = _op_output_tids(cop, cuid, tensors)
            if _is_chunk0_select_scatter(cop) and outs:
                return outs[0], []
            new_outs = [ot for ot in outs if ot not in bfs_seen]
            bfs_seen.update(new_outs)
            nxt_q.extend(new_outs)
    return None, nxt_q


def _find_first_select_scatter(
    dm0_mul_tid: Optional[str],
    consumers_of: Dict[str, List[str]],
    ops: Dict[str, Any],
    tensors: Dict[str, Any],
) -> Optional[str]:
    """BFS forward from ``dm0_mul_tid`` to locate the first ``select_scatter(dim=2, index=0)`` output."""
    if not dm0_mul_tid:
        return None
    bfs_q = [dm0_mul_tid]
    bfs_seen: Set[str] = {dm0_mul_tid}
    for _ in range(12):
        if not bfs_q:
            break
        found, bfs_q = _expand_bfs_frontier(bfs_q, bfs_seen, consumers_of, ops, tensors)
        if found is not None:
            return found
    return None


def _first_consumer_output(
    tid: str, consumers_of: Dict[str, List[str]], ops: Dict[str, Any], tensors: Dict[str, Any], pred,
) -> Optional[str]:
    """Return the first output tensor ID of a consumer of ``tid`` satisfying ``pred(op)``."""
    for cuid in consumers_of.get(tid, []):
        op = ops[cuid]
        if pred(op):
            outs = _op_output_tids(op, cuid, tensors)
            if outs:
                return outs[0]
    return None


def _is_slice_dim2(op: Dict[str, Any]) -> bool:
    """Return True if ``op`` is ``aten::slice(..., dim=2)``."""
    if op.get("op_type") != "aten::slice":
        return False
    sargs = _op_args(op)
    ok, sdim = _arg_scalar(sargs[1]) if len(sargs) >= 2 else (False, None)
    return bool(ok and sdim == 2)


def _follow_core_attn_out_tail(
    out_5d_tid: str,
    v_dtype: Optional[str],
    consumers_of: Dict[str, List[str]],
    ops: Dict[str, Any],
    tensors: Dict[str, Any],
) -> Optional[str]:
    """Walk ``view -> slice(dim=2) -> transpose(1,2) -> [clone]* -> [_to_copy to v_dtype]``."""
    view_out = _first_consumer_output(out_5d_tid, consumers_of, ops, tensors, lambda o: o.get("op_type") in _VIEW_OPS)
    if not view_out:
        return None
    slice_out = _first_consumer_output(view_out, consumers_of, ops, tensors, _is_slice_dim2)
    if not slice_out:
        return None
    trans_out = _first_consumer_output(slice_out, consumers_of, ops, tensors, lambda o: _is_transpose_dims(o, 1, 2))
    if not trans_out:
        return None
    core_out_tid = trans_out
    while True:
        c_list = consumers_of.get(core_out_tid, [])
        if len(c_list) != 1:
            break
        cuid = c_list[0]
        cop = ops[cuid]
        ctype = cop.get("op_type")
        outs = _op_output_tids(cop, cuid, tensors)
        if ctype in ("aten::clone", "aten::contiguous") and outs:
            core_out_tid = outs[0]
            continue
        if ctype == "aten::_to_copy" and outs and _tensor_dtype(outs[0], tensors) == v_dtype:
            core_out_tid = outs[0]
        break
    return core_out_tid


def _unwrap_qk_norm_and_expand(
    q_after: str, k_after: str, q_entry: List[str], k_entry: List[str],
    producer_of: Dict[str, str], ops: Dict[str, Any], tensors: Dict[str, Any],
) -> Tuple[str, str, List[str], List[str], bool, float]:
    """Unwrap optional paired L2-norm and GQA head-expansion ops on ``q`` and ``k``."""
    q_norm = _unwrap_l2norm(q_after, producer_of, ops)
    k_norm = _unwrap_l2norm(k_after, producer_of, ops)
    if q_norm and k_norm and math.isclose(q_norm[1], k_norm[1], rel_tol=1e-6, abs_tol=1e-12):
        qk_l2norm, eps = True, q_norm[1]
        q_pre, k_pre = q_norm[0], k_norm[0]
        q_entry, k_entry = q_entry + q_norm[2], k_entry + k_norm[2]
    else:
        qk_l2norm, eps, q_pre, k_pre = False, 1e-6, q_after, k_after

    q_exp = _unwrap_head_expand(q_pre, producer_of, ops, tensors)
    k_exp = _unwrap_head_expand(k_pre, producer_of, ops, tensors)
    if q_exp and k_exp and q_exp[1] == k_exp[1]:
        return q_exp[0], k_exp[0], q_entry + q_exp[2], k_entry + k_exp[2], qk_l2norm, eps
    return q_pre, k_pre, q_entry, k_entry, qk_l2norm, eps


def _match_chunk_gated_delta_rule(
    cumsum_uid: str,
    ops: Dict[str, Any],
    tensors: Dict[str, Any],
    producer_of: Dict[str, str],
    consumers_of: Dict[str, List[str]],
    order_idx: Dict[str, int],
) -> Optional[Dict[str, Any]]:
    """Attempt to match one unrolled ``chunk_gated_delta_rule`` subgraph anchored at ``cumsum_uid``."""
    cumsum_op = ops[cumsum_uid]
    cargs = _op_args(cumsum_op)
    if len(cargs) < 2:
        return None
    g_chunked_tid = _arg_tensor_id(cargs[0])
    ok_d, cdim = _arg_scalar(cargs[1])
    g_shape = _tensor_shape(g_chunked_tid, tensors) if g_chunked_tid else None
    if not (g_chunked_tid and ok_d and cdim in (-1, 3) and g_shape and len(g_shape) == 4):
        return None
    if int(g_shape[2]) <= 0 or int(g_shape[3]) <= 0:
        return None

    gv_uid = producer_of.get(g_chunked_tid)
    g_padded_tid = _arg_tensor_id(_op_args(ops[gv_uid])[0]) if gv_uid and ops[gv_uid].get("op_type") in _VIEW_OPS else None
    g_unwrapped = _unwrap_pad_transpose(g_padded_tid, producer_of, ops) if g_padded_tid else None
    g_cumsum_outs = _op_output_tids(cumsum_op, cumsum_uid, tensors)
    decay_mask_tid = _find_decay_mask(g_cumsum_outs[0], consumers_of, ops, tensors) if g_cumsum_outs else None
    if not (g_unwrapped and decay_mask_tid):
        return None
    g_in_tid, g_entry_ops = g_unwrapped

    kb = _find_k_beta_padded(decay_mask_tid, consumers_of, producer_of, ops)
    if not kb:
        return None
    k_padded_tid, beta_padded_tid = kb
    k_unwrapped = _unwrap_pad_transpose(k_padded_tid, producer_of, ops)
    beta_unwrapped = _unwrap_pad_transpose(beta_padded_tid, producer_of, ops)
    v_padded_tid = _find_v_padded(beta_padded_tid, k_padded_tid, consumers_of, ops, tensors)
    v_unwrapped = _unwrap_pad_transpose(v_padded_tid, producer_of, ops) if v_padded_tid else None
    q_found = _find_q_padded_and_scale(decay_mask_tid, consumers_of, producer_of, ops, tensors)
    q_unwrapped = _unwrap_pad_transpose(q_found[0], producer_of, ops) if q_found else None
    if not (k_unwrapped and beta_unwrapped and v_unwrapped and q_unwrapped):
        return None

    q_in_tid, k_in_tid, q_entry_ops, k_entry_ops, qk_l2norm, eps = _unwrap_qk_norm_and_expand(
        q_unwrapped[0], k_unwrapped[0], q_unwrapped[1], k_unwrapped[1], producer_of, ops, tensors,
    )
    q_in_shape = _tensor_shape(q_in_tid, tensors)
    if not q_in_shape or len(q_in_shape) != 4:
        return None
    scale_val = q_found[1] if q_found[1] is not None else int(q_in_shape[3]) ** -0.5

    out_5d_first = _find_first_select_scatter(q_found[2], consumers_of, ops, tensors)
    out_5d_tid = _find_select_scatter_tail(out_5d_first, consumers_of, ops, tensors, dim=2) if out_5d_first else None
    core_out_tid = _follow_core_attn_out_tail(out_5d_tid, _tensor_dtype(v_unwrapped[0], tensors), consumers_of, ops, tensors) if out_5d_tid else None
    out_prod_uid = producer_of.get(core_out_tid) if core_out_tid else None
    if not (core_out_tid and out_prod_uid and out_prod_uid in order_idx):
        return None

    entry_ops = set(g_entry_ops + k_entry_ops + beta_unwrapped[1] + v_unwrapped[1] + q_entry_ops)
    return {
        "q_in": q_in_tid, "k_in": k_in_tid, "v_in": v_unwrapped[0], "g_in": g_in_tid,
        "beta_in": beta_unwrapped[0], "out_tid": core_out_tid, "out_prod_uid": out_prod_uid,
        "qk_l2norm": qk_l2norm, "eps": eps, "scale": scale_val, "entry_ops": entry_ops,
        "min_idx": min(order_idx[u] for u in entry_ops if u in order_idx),
        "max_idx": order_idx[out_prod_uid], "cumsum_uid": cumsum_uid,
    }


def _next_fused_gdr_index(ops: Dict[str, Any]) -> int:
    """Return the next free numeric suffix for ``custom.gated_delta_rule::<idx>``."""
    existing_idx = 0
    for uid in ops:
        if uid.startswith("custom.gated_delta_rule::"):
            try:
                existing_idx = max(existing_idx, int(uid.split("::")[-1]) + 1)
            except ValueError:
                pass
    return existing_idx


def _build_fused_gdr_op(
    m: Dict[str, Any], new_uid: str, ops: Dict[str, Any], tensors: Dict[str, Any],
) -> Dict[str, Any]:
    """Construct the ``custom::gated_delta_rule`` op record replacing matched subgraph ``m``."""
    in_tids = [m["q_in"], m["k_in"], m["v_in"], m["g_in"], m["beta_in"]]
    out_tid = m["out_tid"]
    in_dtypes = [_tensor_dtype(t, tensors) or "float32" for t in in_tids]
    device = (ops.get(m["out_prod_uid"]) or {}).get("device") or (tensors.get(out_tid) or {}).get("device") or "cuda"
    new_op: Dict[str, Any] = {
        "op_uid": new_uid,
        "op_type": FUSED_OP,
        "input_tensor_ids": in_tids,
        "output_tensor_ids": [out_tid],
        "input_shapes": [_tensor_shape(t, tensors) or [] for t in in_tids],
        "output_shapes": [_tensor_shape(out_tid, tensors) or []],
        "input_dtypes": in_dtypes,
        "output_dtypes": [_tensor_dtype(out_tid, tensors) or in_dtypes[2]],
        "device": device,
        "attributes": {
            "args": [{"type": "tensor", "tensor_id": t} for t in in_tids],
            "kwargs": {"qk_l2norm": m["qk_l2norm"], "eps": m["eps"], "scale": m["scale"]},
            "qk_l2norm": m["qk_l2norm"], "eps": m["eps"], "scale": m["scale"],
        },
    }
    parent_module = (ops.get(m["cumsum_uid"]) or {}).get("parent_module")
    if parent_module:
        new_op["parent_module"] = parent_module
    return new_op


def _expand_dead_cone_backward(
    cone: Set[str], min_idx: int, protected_inputs: Set[str], replaced_out_prods: Set[str],
    ops: Dict[str, Any], tensors: Dict[str, Any], producer_of: Dict[str, str],
    consumers_of: Dict[str, List[str]], order_idx: Dict[str, int], graph_outputs: Set[str],
) -> None:
    """Pull internal constant/helper producers feeding only ``cone`` into ``cone``."""
    changed = True
    while changed:
        changed = False
        for u in list(cone):
            op = ops.get(u)
            if not op:
                continue
            for itid in _op_input_tids(op):
                if itid in protected_inputs:
                    continue
                pu = producer_of.get(itid)
                if not (pu and pu not in cone and pu not in replaced_out_prods):
                    continue
                pop = ops.get(pu)
                if not (pop and order_idx.get(pu, -1) >= min_idx):
                    continue
                pouts = _op_output_tids(pop, pu, tensors)
                if pouts and all(all(c in cone for c in consumers_of.get(pt, [])) and pt not in graph_outputs for pt in pouts):
                    cone.add(pu)
                    changed = True


def _collect_match_dead_cone(
    m: Dict[str, Any], protected_inputs: Set[str], replaced_out_prods: Set[str],
    ops: Dict[str, Any], tensors: Dict[str, Any], producer_of: Dict[str, str],
    consumers_of: Dict[str, List[str]], order_idx: Dict[str, int], graph_outputs: Set[str],
) -> Set[str]:
    """Collect forward and backward candidate-dead ops belonging solely to matched subgraph ``m``."""
    out_tid = m["out_tid"]
    cone: Set[str] = set(m["entry_ops"])
    queue = list(m["entry_ops"])
    while queue:
        u = queue.pop()
        op = ops.get(u)
        if not op:
            continue
        for otid in _op_output_tids(op, u, tensors):
            if otid == out_tid:
                continue
            for cu in consumers_of.get(otid, []):
                if cu not in cone:
                    cone.add(cu)
                    queue.append(cu)
    _expand_dead_cone_backward(
        cone, m["min_idx"], protected_inputs, replaced_out_prods,
        ops, tensors, producer_of, consumers_of, order_idx, graph_outputs,
    )
    return cone


def _enqueue_newly_dead_producers(
    uid: str, op: Dict[str, Any], live_consumers: Dict[str, Set[str]],
    producer_of: Dict[str, str], all_candidate_dead: Set[str],
    dead_ops: Set[str], worklist: List[str], workset: Set[str],
) -> None:
    """Release ``uid`` from its inputs' consumer sets and queue newly-dead producers."""
    for itid in _op_input_tids(op):
        cs = live_consumers.get(itid)
        if cs is None:
            continue
        cs.discard(uid)
        if cs:
            continue
        pu = producer_of.get(itid)
        if pu and pu in all_candidate_dead and pu not in dead_ops and pu not in workset:
            worklist.append(pu)
            workset.add(pu)


def _sweep_dead_gdr_ops(
    new_order: List[str], all_candidate_dead: Set[str], ops: Dict[str, Any],
    tensors: Dict[str, Any], producer_of: Dict[str, str], graph_outputs: Set[str],
    n_matches: int, orig_len: int,
) -> List[str]:
    """Iteratively remove dead candidate ops and their unreferenced intermediate tensors."""
    live_consumers: Dict[str, Set[str]] = {}
    for uid in new_order:
        op = ops.get(uid)
        if op:
            for itid in _op_input_tids(op):
                live_consumers.setdefault(itid, set()).add(uid)

    dead_ops: Set[str] = set()
    worklist = [u for u in all_candidate_dead if u in ops]
    workset = set(worklist)
    while worklist:
        uid = worklist.pop()
        workset.discard(uid)
        if uid in dead_ops or uid not in ops:
            continue
        op = ops[uid]
        if any(ot in graph_outputs or live_consumers.get(ot) for ot in _op_output_tids(op, uid, tensors)):
            continue
        dead_ops.add(uid)
        _enqueue_newly_dead_producers(uid, op, live_consumers, producer_of, all_candidate_dead, dead_ops, worklist, workset)

    for uid in dead_ops:
        op = ops.pop(uid, None)
        if op:
            for ot in _op_output_tids(op, uid, tensors):
                if ot not in graph_outputs and not live_consumers.get(ot):
                    tensors.pop(ot, None)

    for tid, meta in tensors.items():
        if isinstance(meta, dict) and "consumer_op_uids" in meta:
            meta["consumer_op_uids"] = sorted(live_consumers.get(tid, ()))

    kept = [uid for uid in new_order if uid not in dead_ops]
    logger.info(
        "[GatedDeltaRule Fusion] Fused %d layer(s) into %s: %d -> %d ops (-%d ops)",
        n_matches, FUSED_OP, orig_len, len(kept), orig_len - len(kept),
    )
    return kept


def detect_and_fuse_gated_delta_rule(dag: Dict[str, Any]) -> Dict[str, Any]:
    """Detect and fuse unrolled ``chunk_gated_delta_rule`` subgraphs in ``dag`` in place.

    Replaces each matched subgraph with a single dynamic ``custom::gated_delta_rule``
    op taking ``(q, k, v, g, beta)`` in ``[B, L, Hk/Hv, D]`` / ``[B, L, Hv]`` layout
    and producing ``core_attn_out`` in ``[B, L, Hv, dv]`` at ``v``'s dtype.
    """
    ops: Dict[str, Any] = dag.get("ops") or {}
    tensors: Dict[str, Any] = dag.get("tensors") or {}
    order: List[str] = dag.get("execution_order") or []
    if not ops or not order:
        return dag

    cumsum_uids = [uid for uid in order if (ops.get(uid) or {}).get("op_type") == "aten::cumsum"]
    if not cumsum_uids:
        return dag

    producer_of, consumers_of, order_idx = _build_indices(ops, tensors, order)
    graph_outputs: Set[str] = set(dag.get("output_tensor_ids") or [])
    matches = [
        m for cuid in cumsum_uids
        if (m := _match_chunk_gated_delta_rule(cuid, ops, tensors, producer_of, consumers_of, order_idx)) is not None
    ]
    if not matches:
        return dag

    existing_idx = _next_fused_gdr_index(ops)
    all_candidate_dead: Set[str] = set()
    new_ops_at: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    replaced_out_prods: Set[str] = set()
    protected_inputs: Set[str] = set()

    for m in matches:
        protected_inputs.update((m["q_in"], m["k_in"], m["v_in"], m["g_in"], m["beta_in"]))
        new_uid = f"custom.gated_delta_rule::{existing_idx}"
        existing_idx += 1
        new_ops_at[m["out_prod_uid"]] = (new_uid, _build_fused_gdr_op(m, new_uid, ops, tensors))
        replaced_out_prods.add(m["out_prod_uid"])
        all_candidate_dead.update(_collect_match_dead_cone(
            m, protected_inputs, replaced_out_prods, ops, tensors, producer_of, consumers_of, order_idx, graph_outputs,
        ))

    new_order: List[str] = []
    for uid in order:
        if uid in new_ops_at:
            new_uid, new_op = new_ops_at[uid]
            ops[new_uid] = new_op
            out_tid = new_op["output_tensor_ids"][0]
            if out_tid in tensors:
                tensors[out_tid]["producer_op_uid"] = new_uid
            new_order.append(new_uid)
            del ops[uid]
        else:
            new_order.append(uid)

    dag["execution_order"] = _sweep_dead_gdr_ops(
        new_order, all_candidate_dead, ops, tensors, producer_of, graph_outputs, len(matches), len(order),
    )
    return dag
