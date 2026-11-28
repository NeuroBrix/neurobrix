"""Choice-score MoE routing parser and grouped_mm stacked-expert fusion pass.

Extracted from ``moe_fusion.py`` to keep choice-score routing (sigmoid/softmax, optional
choice bias, optional group mask, gather, renorm, scale) and ``aten::_grouped_mm`` stacked
expert fusion cohesive and modular.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

_log = logging.getLogger(__name__)

_SCORE_VIEW = frozenset({"aten::view", "aten::reshape", "aten::_unsafe_view", "aten::clone", "aten::_to_copy"})
_RESHAPE = frozenset({"aten::view", "aten::reshape", "aten::_unsafe_view", "aten::contiguous", "aten::clone"})
_VIEW_LIKE = _RESHAPE | frozenset({"aten::unsqueeze", "aten::squeeze", "aten::expand", "aten::slice", "aten::select"})
_GROUP_STOP = frozenset({"aten::mm", "aten::addmm", "aten::bmm", "aten::_grouped_mm", "aten::linear", "aten::embedding"})


class ChoiceRoutingPlan(NamedTuple):
    """Parsed choice-score routing cluster read from the graph (zero hardcode)."""
    logits_tid: str
    score_func: str
    choice_bias_tid: Optional[str]
    n_group: int
    topk_group: int
    group_topk_k: int
    mask_fill_value: float
    top_k: int
    num_experts: int
    norm_topk_prob: bool
    norm_eps: float
    routed_scaling_factor: float
    indices_tid: str
    weights_tid: str
    routing_ops: Set[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score_func": self.score_func,
            "choice_bias_tid": self.choice_bias_tid,
            "n_group": self.n_group,
            "topk_group": self.topk_group,
            "group_topk_k": self.group_topk_k,
            "mask_fill_value": self.mask_fill_value,
            "norm_eps": self.norm_eps,
            "routed_scaling_factor": self.routed_scaling_factor,
        }


def _tshape(tensors: Dict[str, Any], tid: Optional[str]) -> Optional[List[int]]:
    """Return tensor shape list or None."""
    if not tid:
        return None
    sh = (tensors.get(tid) or {}).get("shape")
    return list(sh) if sh is not None else None


def _same_sh(tensors: Dict[str, Any], a: Optional[str], b: Optional[str]) -> bool:
    """Return True if tensors ``a`` and ``b`` have identical non-empty shapes."""
    sa, sb = _tshape(tensors, a), _tshape(tensors, b)
    return bool(sa and sb and sa == sb)


def _in_tids(op: Dict[str, Any]) -> List[str]:
    """Collect tensor IDs referenced in ``op`` attributes.args (fallback: input_tensor_ids)."""
    raw = (op.get("attributes") or {}).get("args")
    if not isinstance(raw, list):
        return list(op.get("input_tensor_ids") or [])
    items = [sub for a in raw for sub in (a if isinstance(a, list) else (a,))]
    out = [x["tensor_id"] for x in items if isinstance(x, dict) and x.get("type") in ("tensor", "tensor_ref") and x.get("tensor_id")]
    return out or list(op.get("input_tensor_ids") or [])


def _in0(op: Dict[str, Any]) -> Optional[str]:
    """Return first input tensor ID of ``op``."""
    ins = _in_tids(op)
    return ins[0] if ins else None


def _sargs(op: Dict[str, Any]) -> List[Any]:
    """Return scalar (non-tensor) positional arguments of ``op``."""
    out: List[Any] = []
    for a in (op.get("attributes") or {}).get("args", []):
        if isinstance(a, dict) and a.get("type") in ("tensor", "tensor_ref"):
            continue
        if isinstance(a, dict) and "value" in a:
            out.append(a["value"])
        elif not isinstance(a, dict):
            out.append(a)
    return out


def _topk_k(op: Dict[str, Any]) -> Optional[int]:
    """Extract integer k from an ``aten::topk`` op."""
    attrs = op.get("attributes") or {}
    if "k" in attrs and isinstance(attrs["k"], int):
        return int(attrs["k"])
    sa = _sargs(op)
    return int(sa[0]) if sa and isinstance(sa[0], int) else None


def _topk_largest(op: Dict[str, Any]) -> bool:
    """Return True unless ``aten::topk`` explicitly specifies ``largest=False``."""
    attrs = op.get("attributes") or {}
    if "largest" in attrs:
        return bool(attrs["largest"])
    sa = _sargs(op)
    return bool(sa[2]) if len(sa) > 2 and isinstance(sa[2], bool) else True


def _is_param_tid(tensors: Dict[str, Any], tid: Optional[str]) -> bool:
    """Return True if ``tid`` refers to a graph parameter tensor."""
    if not tid:
        return False
    meta = tensors.get(tid) or {}
    return bool(tid.startswith("param::") or meta.get("is_parameter"))


def _walk_choice_score_chain(ops, tensors, producer_map, topk_in: str, E: int, routing_ops: Set[str], declined_cls):
    """Walk backward from ``topk_in`` through views, optional masked_fill, optional choice_bias add, and score op."""
    chain: Set[str] = {topk_in}
    bias_tid: Optional[str] = None
    mask_tid: Optional[str] = None
    mask_fill_uid: Optional[str] = None
    mask_fill_value: float = 0.0
    tid = topk_in
    while True:
        pu = producer_map.get(tid)
        p = ops.get(pu) or {}
        t, ins = p.get("op_type"), _in_tids(p)
        if t in _SCORE_VIEW and ins:
            routing_ops.add(pu)
            tid = ins[0]
        elif t in ("aten::masked_fill", "aten::masked_fill_") and len(ins) >= 2 and mask_tid is None:
            sa = _sargs(p)
            if not sa or not isinstance(sa[0], (int, float)):
                return declined_cls(f"masked_fill {pu} has no scalar fill value")
            mask_fill_value, mask_fill_uid, mask_tid, tid = float(sa[0]), pu, ins[1], ins[0]
            routing_ops.add(pu)
        elif t == "aten::add" and len(ins) == 2 and bias_tid is None and any(_tshape(tensors, b) == [E] for b in ins):
            bias_tid = next(b for b in ins if _tshape(tensors, b) == [E])
            tid = next((a for a in ins if a != bias_tid), ins[0])
            routing_ops.add(pu)
        else:
            break
        chain.add(tid)

    score_uid = producer_map.get(tid)
    score_op = ops.get(score_uid) or {}
    st = score_op.get("op_type")
    if st == "aten::sigmoid":
        score_func = "sigmoid"
    elif st == "aten::_softmax":
        sm_args = _sargs(score_op)
        if not sm_args or sm_args[0] not in (-1, 1):
            return declined_cls(f"the choice softmax is not over the expert axis ({sm_args})")
        score_func = "softmax"
    else:
        return declined_cls(f"unsupported choice score function {st!r}")
    logits_tid = _in0(score_op)
    if logits_tid is None or not _same_sh(tensors, logits_tid, tid):
        return declined_cls("the choice score op input does not match the [tokens, experts] shape")
    routing_ops.add(score_uid)
    return (logits_tid, tid, score_func, bias_tid, mask_tid, mask_fill_uid, mask_fill_value, chain)


def _collect_group_mask_ops(ops, producer_map, mask_tid: str, chain: Set[str], declined_cls):
    """Collect the group-mask subgraph feeding ``mask_tid`` up to ``chain``."""
    group_ops: Set[str] = set()
    seen_t: Set[str] = set(chain)
    todo = [mask_tid]
    while todo:
        m = todo.pop()
        if m in seen_t:
            continue
        seen_t.add(m)
        gu = producer_map.get(m)
        g = ops.get(gu) or {}
        if not g:
            return declined_cls(f"group mask tensor {m} has no producer op")
        if g.get("op_type") in _GROUP_STOP:
            return declined_cls(f"group mask walks past {g.get('op_type')!r} at {gu}")
        group_ops.add(gu)
        todo.extend(_in_tids(g))
    return group_ops


def _parse_group_mask_subgraph(ops, tensors, consumer_map, producer_map, mask_tid: str, mask_fill_uid: str, chain: Set[str], E: int, declined_cls):
    """Validate optional group-mask subgraph and return ``(n_group, topk_group, group_topk_k, group_ops)``."""
    group_ops = _collect_group_mask_ops(ops, producer_map, mask_tid, chain, declined_cls)
    if isinstance(group_ops, declined_cls):
        return group_ops
    if not any(ops[u].get("op_type") in ("aten::bitwise_not", "aten::logical_not") for u in group_ops):
        return declined_cls("group mask has no bitwise_not/logical_not inversion before masked_fill")

    per_group_topk, select_group_topk = None, None
    for u in (u for u in group_ops if ops[u].get("op_type") == "aten::topk"):
        gsh = _tshape(tensors, _in0(ops[u]))
        if gsh and len(gsh) == 3:
            per_group_topk = u
        elif gsh and len(gsh) == 2:
            select_group_topk = u
        else:
            return declined_cls(f"unexpected group top-k shape {gsh} at {u}")

    if select_group_topk is None or not _topk_largest(ops[select_group_topk]):
        return declined_cls("group mask has no valid 2D group-selection top-k")
    n_group = int(_tshape(tensors, _in0(ops[select_group_topk]))[1])
    topk_group = _topk_k(ops[select_group_topk])
    if topk_group is None or topk_group < 1 or topk_group > n_group or n_group < 1 or E % n_group != 0:
        return declined_cls(f"invalid group parameters n_group={n_group}, topk_group={topk_group}, E={E}")

    group_topk_k = _parse_per_group_reducer(ops, tensors, group_ops, per_group_topk, n_group, E, declined_cls)
    if isinstance(group_topk_k, declined_cls):
        return group_topk_k

    allowed = group_ops | {mask_fill_uid}
    for u in group_ops:
        for ot in ops[u].get("output_tensor_ids", []):
            esc = [c for c in consumer_map.get(ot, []) if c not in allowed]
            if esc:
                return declined_cls(f"group mask tensor {ot} escapes to {esc[0]}")
    return n_group, topk_group, group_topk_k, group_ops


def _parse_per_group_reducer(ops, tensors, group_ops: Set[str], per_group_topk: Optional[str], n_group: int, E: int, declined_cls):
    """Extract ``group_topk_k`` from either a 3D per-group top-k or an amax/max reducer."""
    if per_group_topk is not None:
        if not _topk_largest(ops[per_group_topk]):
            return declined_cls("per-group top-k has largest=False")
        pg_sh = _tshape(tensors, _in0(ops[per_group_topk]))
        if int(pg_sh[1]) != n_group or int(pg_sh[2]) != E // n_group:
            return declined_cls(f"per-group top-k shape {pg_sh} disagrees with n_group={n_group}, E={E}")
        gtk = _topk_k(ops[per_group_topk])
        if gtk is None or gtk < 1 or gtk > E // n_group:
            return declined_cls(f"invalid group_topk_k={gtk}")
        return gtk
    amax_ops = [u for u in group_ops if ops[u].get("op_type") in ("aten::amax", "aten::max")]
    if len(amax_ops) != 1:
        return declined_cls("group mask has neither a 3D per-group top-k nor an amax/max")
    ash = _tshape(tensors, _in0(ops[amax_ops[0]]))
    if not ash or len(ash) != 3 or int(ash[1]) != n_group or int(ash[2]) != E // n_group:
        return declined_cls(f"group amax shape {ash} disagrees with n_group={n_group}, E={E}")
    return 1


def _parse_gather_and_post_weights(ops, consumer_map, scores_tid: str, indices_tid: str, routing_ops: Set[str], declined_cls):
    """Parse ``gather(scores, 1, indices) -> [renorm] -> [scale] -> [cast]`` and return ``(w_tid, renorm, norm_eps, scale)``."""
    gathers = [u for u in consumer_map.get(scores_tid, []) if ops.get(u, {}).get("op_type") == "aten::gather"]
    if len(gathers) != 1:
        return declined_cls(f"expected 1 aten::gather on unbiased scores {scores_tid}, found {len(gathers)}")
    gu = gathers[0]
    g_ins, g_sa = _in_tids(ops[gu]), _sargs(ops[gu])
    g_dim = ops[gu].get("attributes", {}).get("dim", g_sa[0] if g_sa else None)
    if len(g_ins) < 2 or g_ins[0] != scores_tid or g_ins[1] != indices_tid or g_dim not in (1, -1):
        return declined_cls(f"gather {gu} does not gather unbiased scores at topk indices along dim=1")
    routing_ops.add(gu)
    w_tid = ops[gu]["output_tensor_ids"][0]

    renorm, norm_eps = False, 0.0
    wcs = consumer_map.get(w_tid, [])
    if sorted(ops.get(u, {}).get("op_type", "") for u in wcs) == ["aten::div", "aten::sum"]:
        res = _parse_renorm_pair(ops, consumer_map, w_tid, wcs, routing_ops, declined_cls)
        if isinstance(res, declined_cls):
            return res
        w_tid, norm_eps = res
        renorm = True

    scale = 1.0
    wcs = consumer_map.get(w_tid, [])
    if len(wcs) == 1 and ops.get(wcs[0], {}).get("op_type") == "aten::mul":
        mu = wcs[0]
        m_sa = _sargs(ops[mu])
        if _in_tids(ops[mu]) == [w_tid] and len(m_sa) == 1 and isinstance(m_sa[0], (int, float)):
            scale = float(m_sa[0])
            routing_ops.add(mu)
            w_tid = ops[mu]["output_tensor_ids"][0]

    wcs = consumer_map.get(w_tid, [])
    if len(wcs) == 1 and ops.get(wcs[0], {}).get("op_type") == "aten::_to_copy":
        routing_ops.add(wcs[0])
        w_tid = ops[wcs[0]]["output_tensor_ids"][0]
    return w_tid, renorm, norm_eps, scale


def _parse_renorm_pair(ops, consumer_map, w_tid: str, wcs: List[str], routing_ops: Set[str], declined_cls):
    """Parse ``w / (sum(w, dim=-1, keepdim=True) [+ eps])`` and return ``(new_w_tid, norm_eps)``."""
    sum_uid = next(u for u in wcs if ops[u]["op_type"] == "aten::sum")
    div_uid = next(u for u in wcs if ops[u]["op_type"] == "aten::div")
    sa, sattrs = _sargs(ops[sum_uid]), ops[sum_uid].get("attributes", {})
    dims = sattrs.get("dim", sa[0] if sa else None)
    keep = sattrs.get("keepdim", sa[1] if len(sa) > 1 else False)
    if dims not in ([-1], [1], -1, 1) or keep is not True:
        return declined_cls(f"gathered weights summed over {dims} keepdim={keep}, not renormalised")
    sum_out = ops[sum_uid]["output_tensor_ids"][0]
    sum_cs = consumer_map.get(sum_out, [])
    norm_eps = 0.0
    if sum_cs == [div_uid]:
        denom_tid = sum_out
        routing_ops |= {sum_uid, div_uid}
    elif len(sum_cs) == 1 and ops.get(sum_cs[0], {}).get("op_type") == "aten::add":
        eps_uid = sum_cs[0]
        eps_sa = _sargs(ops[eps_uid])
        if _in_tids(ops[eps_uid]) != [sum_out] or not eps_sa or not isinstance(eps_sa[0], (int, float)):
            return declined_cls(f"renorm denominator add {eps_uid} is not sum + scalar_eps")
        denom_tid = ops[eps_uid]["output_tensor_ids"][0]
        if consumer_map.get(denom_tid, []) != [div_uid]:
            return declined_cls(f"renorm denominator {denom_tid} does not feed only {div_uid}")
        norm_eps = float(eps_sa[0])
        routing_ops |= {sum_uid, eps_uid, div_uid}
    else:
        return declined_cls(f"renorm sum {sum_out} feeds unexpected ops {sum_cs}")
    if _in_tids(ops[div_uid])[:2] != [w_tid, denom_tid]:
        return declined_cls(f"renorm div {div_uid} inputs do not match (weights, denom)")
    return ops[div_uid]["output_tensor_ids"][0], norm_eps


def parse_choice_routing(ops, tensors, consumer_map, producer_map, topk_uid: str, declined_cls):
    """Parse a choice-score routing cluster around ``topk_uid`` into a ``ChoiceRoutingPlan`` or ``declined_cls``."""
    topk = ops[topk_uid]
    k = _topk_k(topk)
    if k is None or k <= 1:
        return declined_cls("the choice top-k selects at most one expert")
    topk_in = _in0(topk)
    tsh = _tshape(tensors, topk_in)
    if not tsh or len(tsh) != 2:
        return declined_cls(f"the choice top-k input is not a rank-2 [tokens, experts] tensor ({tsh})")
    E = int(tsh[1])
    tk_args = _sargs(topk)
    if (len(tk_args) > 1 and tk_args[1] not in (-1, 1)) or not _topk_largest(topk):
        return declined_cls("the choice top-k does not select largest scores over the expert axis")
    outs = topk.get("output_tensor_ids", [])
    if len(outs) < 2 or consumer_map.get(outs[0]):
        return declined_cls("the choice top-k has invalid (values, indices) outputs")
    indices_tid = outs[1]

    routing_ops: Set[str] = {topk_uid}
    sc = _walk_choice_score_chain(ops, tensors, producer_map, topk_in, E, routing_ops, declined_cls)
    if isinstance(sc, declined_cls):
        return sc
    logits_tid, scores_tid, score_func, bias_tid, mask_tid, mask_fill_uid, mask_fill_value, chain = sc

    n_group, topk_group, group_topk_k = 1, 1, 1
    if mask_tid is not None:
        gm = _parse_group_mask_subgraph(ops, tensors, consumer_map, producer_map, mask_tid, mask_fill_uid, chain, E, declined_cls)
        if isinstance(gm, declined_cls):
            return gm
        n_group, topk_group, group_topk_k, group_ops = gm
        routing_ops.update(group_ops)

    gw = _parse_gather_and_post_weights(ops, consumer_map, scores_tid, indices_tid, routing_ops, declined_cls)
    if isinstance(gw, declined_cls):
        return gw
    w_tid, renorm, norm_eps, scale = gw

    for u in routing_ops:
        for ot in ops[u].get("output_tensor_ids", []):
            if ot not in (indices_tid, w_tid):
                esc = [c for c in consumer_map.get(ot, []) if c not in routing_ops]
                if esc:
                    return declined_cls(f"choice routing tensor {ot} escapes to {esc[0]}")

    return ChoiceRoutingPlan(
        logits_tid=logits_tid, score_func=score_func, choice_bias_tid=bias_tid,
        n_group=n_group, topk_group=topk_group, group_topk_k=group_topk_k,
        mask_fill_value=mask_fill_value, top_k=k, num_experts=E,
        norm_topk_prob=renorm, norm_eps=norm_eps, routed_scaling_factor=scale,
        indices_tid=indices_tid, weights_tid=w_tid, routing_ops=routing_ops,
    )


def _only_consumer(consumer_map, tid: str) -> Optional[str]:
    cs = consumer_map.get(tid, [])
    return cs[0] if len(cs) == 1 else None


def _parse_sorted_offsets(ops, consumer_map, indices_tid: str, interior: Set[str], declined_cls):
    """Walk ``indices_tid -> view -> sort -> inv_sort`` and ``histc/bincount -> cumsum -> offs_tid``."""
    idx_cs = [c for c in consumer_map.get(indices_tid, []) if c not in interior]
    if len(idx_cs) != 1 or ops.get(idx_cs[0], {}).get("op_type") not in _RESHAPE:
        return declined_cls(f"choice top-k indices {indices_tid} do not feed a single reshape into sorted dispatch")
    interior.add(idx_cs[0])
    flat_idx_tid = ops[idx_cs[0]]["output_tensor_ids"][0]

    s0 = [c for c in consumer_map.get(flat_idx_tid, []) if ops.get(c, {}).get("op_type") == "aten::sort"]
    if len(s0) != 1 or len(ops[s0[0]].get("output_tensor_ids", [])) < 2:
        return declined_cls(f"expected 1 aten::sort on flattened indices {flat_idx_tid}")
    interior.add(s0[0])
    sort0_outs = ops[s0[0]]["output_tensor_ids"]
    perm_tid = sort0_outs[1]

    s1 = [c for c in consumer_map.get(perm_tid, []) if ops.get(c, {}).get("op_type") == "aten::sort"]
    if len(s1) != 1 or len(ops[s1[0]].get("output_tensor_ids", [])) < 2:
        return declined_cls(f"expected 1 inverse aten::sort on perm {perm_tid}")
    interior.add(s1[0])
    inv_perm_tid = ops[s1[0]]["output_tensor_ids"][1]

    offs_f = [c for c in consumer_map.get(perm_tid, []) if ops.get(c, {}).get("op_type") == "aten::index" and flat_idx_tid in _in_tids(ops[c])]
    if len(offs_f) == 1:
        interior.add(offs_f[0])
        cur = ops[offs_f[0]]["output_tensor_ids"][0]
    elif consumer_map.get(sort0_outs[0]):
        cur = sort0_outs[0]
    else:
        return declined_cls("sorted expert ids do not feed histogram/bincount")

    cu = _only_consumer(consumer_map, cur)
    if cu is not None and ops.get(cu, {}).get("op_type") == "aten::_to_copy":
        interior.add(cu)
        cur = ops[cu]["output_tensor_ids"][0]
        cu = _only_consumer(consumer_map, cur)
    if cu is None or ops.get(cu, {}).get("op_type") not in ("aten::histc", "aten::bincount"):
        return declined_cls("expected aten::histc or aten::bincount for expert counts")
    interior.add(cu)
    cu = _only_consumer(consumer_map, ops[cu]["output_tensor_ids"][0])
    if cu is None or ops.get(cu, {}).get("op_type") != "aten::cumsum":
        return declined_cls("expected aten::cumsum for grouped_mm offsets")
    interior.add(cu)
    return perm_tid, inv_perm_tid, ops[cu]["output_tensor_ids"][0]


def _resolve_gmm_slabs(ops, tensors, producer_map, gmm0_uid: str, gmm1_uid: str, E: int, interior: Set[str], declined_cls):
    """Resolve stacked 3D parameter slabs and dimensions ``(H, F, in_axis, out_in_axis)`` for ``gmm0`` and ``gmm1``."""
    g0_ins, g1_ins = _in_tids(ops[gmm0_uid]), _in_tids(ops[gmm1_uid])
    if len(g0_ins) < 3 or len(g1_ins) < 3:
        return declined_cls("aten::_grouped_mm lacks (input, weight, offsets) inputs")
    w_in, w_in_t = _resolve_one_slab(ops, tensors, producer_map, g0_ins[1], E, interior)
    w_out, w_out_t = _resolve_one_slab(ops, tensors, producer_map, g1_ins[1], E, interior)
    if w_in is None or w_out is None or w_in == w_out:
        return declined_cls("grouped_mm weights do not resolve to two distinct 3D [E, ., .] parameter slabs")
    if not tensors.get(w_in, {}).get("dtype"):
        return declined_cls(f"input slab {w_in} carries no dtype in the graph")
    in_sh, out_sh = _tshape(tensors, w_in), _tshape(tensors, w_out)
    in_axis, out_in_axis = (1 if w_in_t else 0), (1 if w_out_t else 0)
    H, two_F = int(in_sh[2] if w_in_t else in_sh[1]), int(in_sh[1] if w_in_t else in_sh[2])
    F, H_out = int(out_sh[2] if w_out_t else out_sh[1]), int(out_sh[1] if w_out_t else out_sh[2])
    if two_F != 2 * F or H_out != H:
        return declined_cls(f"stacked slab shapes {in_sh} (t={w_in_t}) and {out_sh} (t={w_out_t}) do not match H={H}, 2F={2 * F}")
    return g0_ins[0], g1_ins[0], w_in, w_out, H, F, in_axis, out_in_axis


def _resolve_one_slab(ops, tensors, producer_map, w_arg_tid: str, E: int, interior: Set[str]):
    """Resolve an optional ``transpose(-2, -1)`` around a 3D ``[E, ., .]`` parameter tensor."""
    pu = producer_map.get(w_arg_tid)
    p = ops.get(pu) or {}
    transposed, w_tid = False, w_arg_tid
    if p.get("op_type") == "aten::transpose":
        sa = _sargs(p)
        d0 = p.get("attributes", {}).get("dim0", sa[0] if len(sa) > 0 else None)
        d1 = p.get("attributes", {}).get("dim1", sa[1] if len(sa) > 1 else None)
        if {d0, d1} not in ({-2, -1}, {1, 2}):
            return None, None
        interior.add(pu)
        w_tid, transposed = _in0(p), True
    wsh = _tshape(tensors, w_tid) if _is_param_tid(tensors, w_tid) else None
    if not wsh or len(wsh) != 3 or int(wsh[0]) != E:
        return None, None
    return w_tid, transposed


def _walk_sorted_hidden_to_2d(ops, producer_map, sorted_hidden_tid: str, perm_tid: str, interior: Set[str], declined_cls):
    """Walk ``sorted_hidden_tid`` backward through perm index and token-index arange chain to ``hidden_2d_tid``."""
    sh_uid = producer_map.get(sorted_hidden_tid)
    if ops.get(sh_uid, {}).get("op_type") != "aten::index":
        return declined_cls(f"gmm0 input {sorted_hidden_tid} is not produced by aten::index")
    sh_ins = _in_tids(ops[sh_uid])
    if len(sh_ins) < 2 or sh_ins[1] != perm_tid:
        return declined_cls(f"gmm0 input index {sh_uid} is not permuted by perm {perm_tid}")
    interior.add(sh_uid)
    exp_hid_uid = producer_map.get(sh_ins[0])
    if ops.get(exp_hid_uid, {}).get("op_type") != "aten::index":
        return declined_cls(f"expanded hidden {sh_ins[0]} is not produced by aten::index")
    eh_ins = _in_tids(ops[exp_hid_uid])
    if len(eh_ins) < 2:
        return declined_cls(f"expanded hidden index {exp_hid_uid} lacks (hidden, token_ids)")
    interior.add(exp_hid_uid)
    tok_ops = _VIEW_LIKE | {"aten::arange", "aten::clone", "aten::_to_copy", "aten::repeat_interleave", "aten::repeat"}
    todo, seen = [eh_ins[1]], set()
    while todo:
        t_cur = todo.pop()
        if t_cur in seen:
            continue
        seen.add(t_cur)
        pu = producer_map.get(t_cur)
        if pu is None or pu in interior:
            continue
        p = ops.get(pu) or {}
        if p.get("op_type") not in tok_ops:
            return declined_cls(f"unexpected op {p.get('op_type')!r} ({pu}) on token_ids chain")
        interior.add(pu)
        todo.extend(_in_tids(p))
    return eh_ins[0]


def _walk_gmm_combine_exit(ops, tensors, consumer_map, producer_map, weights_tid: str, perm_tid: str, inv_perm_tid: str, gmm1_out: str, hidden_2d_tid: str, k: int, H: int, interior: Set[str], declined_cls):
    """Walk weighted combine, unsort, reshape, sum, optional cast/reshape, and match ``hidden_tid``."""
    w_view_uid = _only_consumer(consumer_map, weights_tid)
    if w_view_uid is None or ops.get(w_view_uid, {}).get("op_type") not in _RESHAPE:
        return declined_cls(f"choice weights {weights_tid} do not feed a single reshape into grouped_mm dispatch")
    interior.add(w_view_uid)
    w_flat_tid = ops[w_view_uid]["output_tensor_ids"][0]
    w_perm_uid = _only_consumer(consumer_map, w_flat_tid)
    if w_perm_uid is None or ops.get(w_perm_uid, {}).get("op_type") != "aten::index" or _in_tids(ops[w_perm_uid])[:2] != [w_flat_tid, perm_tid]:
        return declined_cls(f"flattened weights {w_flat_tid} are not permuted by perm {perm_tid}")
    interior.add(w_perm_uid)
    sorted_w_tid = ops[w_perm_uid]["output_tensor_ids"][0]

    mul_uid = _only_consumer(consumer_map, gmm1_out)
    if mul_uid is None or ops.get(mul_uid, {}).get("op_type") != "aten::mul" or set(_in_tids(ops[mul_uid])) != {gmm1_out, sorted_w_tid}:
        return declined_cls(f"gmm1 output {gmm1_out} is not weighted by sorted routing weights {sorted_w_tid}")
    interior.add(mul_uid)
    cur = ops[mul_uid]["output_tensor_ids"][0]

    unsort_uid = _only_consumer(consumer_map, cur)
    if unsort_uid is None or ops.get(unsort_uid, {}).get("op_type") != "aten::index" or _in_tids(ops[unsort_uid])[:2] != [cur, inv_perm_tid]:
        return declined_cls(f"weighted expert output {cur} is not unpermuted by inv_perm {inv_perm_tid}")
    interior.add(unsort_uid)
    cur = ops[unsort_uid]["output_tensor_ids"][0]

    unflat_uid = _only_consumer(consumer_map, cur)
    if unflat_uid is None or ops.get(unflat_uid, {}).get("op_type") not in _RESHAPE:
        return declined_cls(f"unpermuted expert output {cur} is not reshaped to [T, K, H]")
    ush = _tshape(tensors, ops[unflat_uid]["output_tensor_ids"][0])
    if not ush or len(ush) != 3 or int(ush[1]) != k or int(ush[2]) != H:
        return declined_cls(f"unpermuted reshape shape {ush} does not match [T, K={k}, H={H}]")
    interior.add(unflat_uid)
    sum_uid = _only_consumer(consumer_map, ops[unflat_uid]["output_tensor_ids"][0])
    if sum_uid is None or ops.get(sum_uid, {}).get("op_type") != "aten::sum":
        return declined_cls("reshaped expert output is not summed over top_k")
    sattrs, sa = ops[sum_uid].get("attributes", {}), _sargs(ops[sum_uid])
    if sattrs.get("dim", sa[0] if sa else None) not in ([1], 1):
        return declined_cls("expert combine sum is not over dim=1")
    interior.add(sum_uid)
    exit_tid = ops[sum_uid]["output_tensor_ids"][0]
    return _match_exit_and_hidden_shape(ops, tensors, consumer_map, producer_map, exit_tid, hidden_2d_tid, H, interior, declined_cls)


def _match_exit_and_hidden_shape(ops, tensors, consumer_map, producer_map, exit_tid: str, hidden_2d_tid: str, H: int, interior: Set[str], declined_cls):
    """Absorb optional trailing cast/reshape on ``exit_tid`` and walk ``hidden_2d_tid`` upward to match its shape."""
    cu = _only_consumer(consumer_map, exit_tid)
    if cu is not None and ops.get(cu, {}).get("op_type") == "aten::_to_copy":
        interior.add(cu)
        exit_tid = ops[cu]["output_tensor_ids"][0]
        cu = _only_consumer(consumer_map, exit_tid)
    if cu is not None and ops.get(cu, {}).get("op_type") in _RESHAPE:
        cand = ops[cu]["output_tensor_ids"][0]
        csh = _tshape(tensors, cand) or []
        if _same_sh(tensors, cand, hidden_2d_tid) or (len(csh) == 3 and csh[-1] == H):
            interior.add(cu)
            exit_tid = cand

    t, hidden_tid = hidden_2d_tid, None
    while True:
        if _same_sh(tensors, t, exit_tid):
            hidden_tid = t
            break
        pu = producer_map.get(t)
        if ops.get(pu, {}).get("op_type") not in _RESHAPE or len(consumer_map.get(t, [])) != 1:
            break
        interior.add(pu)
        t = _in0(ops[pu])
    if hidden_tid is None:
        return declined_cls(f"no tensor on the hidden-state chain matches block exit shape {_tshape(tensors, exit_tid)}")
    return hidden_tid, exit_tid


def _verify_gmm_closure(ops, consumer_map, producer_map, interior: Set[str], allowed_in: Set[str], exit_tid: str, declined_cls):
    """Verify no unexpected tensor enters or escapes ``interior``."""
    for uid in interior:
        for tid in _in_tids(ops[uid]):
            if producer_map.get(tid) not in interior and tid not in allowed_in:
                return declined_cls(f"{tid} enters the grouped_mm block at {uid} from outside it")
        for ot in ops[uid].get("output_tensor_ids", []):
            if ot != exit_tid:
                esc = [c for c in consumer_map.get(ot, []) if c not in interior]
                if esc:
                    return declined_cls(f"{ot} escapes the grouped_mm block to {esc[0]}")
    return None


def fuse_grouped_mm_stacked_layer(
    dag, ops, execution_order, tensors, consumer_map, producer_map, topk_uid,
    *, choice_plan: ChoiceRoutingPlan, declared_norm=None, tok,
    declined_cls, split_halves_fn, stacked_fused_op_fn,
):
    """Match and fuse a stacked-expert MoE block dispatched via ``aten::_grouped_mm``."""
    E, k = choice_plan.num_experts, choice_plan.top_k
    interior: Set[str] = set(choice_plan.routing_ops)
    order_pos = {u: i for i, u in enumerate(execution_order)}

    so = _parse_sorted_offsets(ops, consumer_map, choice_plan.indices_tid, interior, declined_cls)
    if isinstance(so, declined_cls):
        return so
    perm_tid, inv_perm_tid, offs_tid = so

    gmm_uids = [c for c in consumer_map.get(offs_tid, []) if ops.get(c, {}).get("op_type") == "aten::_grouped_mm"]
    if len(gmm_uids) != 2 or len(consumer_map.get(offs_tid, [])) != 2:
        return declined_cls(f"expected exactly 2 aten::_grouped_mm consumers of offsets {offs_tid}, got {gmm_uids}")
    gmm0_uid, gmm1_uid = sorted(gmm_uids, key=lambda u: order_pos.get(u, 0))
    interior |= {gmm0_uid, gmm1_uid}

    slabs = _resolve_gmm_slabs(ops, tensors, producer_map, gmm0_uid, gmm1_uid, E, interior, declined_cls)
    if isinstance(slabs, declined_cls):
        return slabs
    sorted_hidden_tid, swiglu_in_tid, w_in, w_out, H, F, in_axis, out_in_axis = slabs

    split_uid = _only_consumer(consumer_map, ops[gmm0_uid]["output_tensor_ids"][0])
    if split_uid is None or ops.get(split_uid, {}).get("op_type") != "aten::split":
        return declined_cls("gmm0 output does not feed a single aten::split")
    halves = split_halves_fn(ops, consumer_map, split_uid, tensors, F)
    if isinstance(halves, declined_cls):
        return halves
    gate_offset, silu_uid, swiglu_mul_uid = halves
    if ops[swiglu_mul_uid]["output_tensor_ids"][0] != swiglu_in_tid:
        return declined_cls("SwiGLU output does not feed gmm1")
    interior |= {split_uid, silu_uid, swiglu_mul_uid}

    hidden_2d_tid = _walk_sorted_hidden_to_2d(ops, producer_map, sorted_hidden_tid, perm_tid, interior, declined_cls)
    if isinstance(hidden_2d_tid, declined_cls):
        return hidden_2d_tid
    ex = _walk_gmm_combine_exit(ops, tensors, consumer_map, producer_map, choice_plan.weights_tid, perm_tid, inv_perm_tid, ops[gmm1_uid]["output_tensor_ids"][0], hidden_2d_tid, k, H, interior, declined_cls)
    if isinstance(ex, declined_cls):
        return ex
    hidden_tid, exit_tid = ex

    allowed_in = {hidden_tid, choice_plan.logits_tid, w_in, w_out} | ({choice_plan.choice_bias_tid} if choice_plan.choice_bias_tid is not None else set())
    err = _verify_gmm_closure(ops, consumer_map, producer_map, interior, allowed_in, exit_tid, declined_cls)
    if err is not None:
        return err
    return _emit_and_place_gmm_op(ops, execution_order, tensors, consumer_map, producer_map, topk_uid, choice_plan, declared_norm, tok, interior, hidden_tid, exit_tid, w_in, w_out, H, F, in_axis, gate_offset, out_in_axis, stacked_fused_op_fn, declined_cls)


def _emit_and_place_gmm_op(
    ops, execution_order, tensors, consumer_map, producer_map, topk_uid,
    choice_plan: ChoiceRoutingPlan, declared_norm, tok, interior: Set[str],
    hidden_tid: str, exit_tid: str, w_in: str, w_out: str,
    H: int, F: int, in_axis: int, gate_offset: int, out_in_axis: int,
    stacked_fused_op_fn, declined_cls,
):
    """Validate norm_topk_prob, build the stacked fused MoE op, and place it in ``execution_order``."""
    if declared_norm is not None and bool(declared_norm) != choice_plan.norm_topk_prob:
        raise RuntimeError(
            f"ZERO FALLBACK: [MoE Fusion] {topk_uid}: the registry declares "
            f"norm_topk_prob={bool(declared_norm)} but the container's graph computes "
            f"norm_topk_prob={choice_plan.norm_topk_prob} — fix the data at its source.")
    parent = ops[topk_uid].get("parent_module", "") or ""
    bm = re.match(rf"({re.escape(tok['layers'])}\.\d+)", parent)
    fused_uid = f"moe_fused::{bm.group(1) if bm else topk_uid}"
    if fused_uid in ops:
        fused_uid = f"moe_fused::{topk_uid}"
    fused_op = stacked_fused_op_fn(
        tensors, fused_uid, top_k=choice_plan.top_k, num_experts=choice_plan.num_experts, F=F, H=H,
        hidden_tid=hidden_tid, gate_scores_tid=choice_plan.logits_tid, in_tid=w_in, out_tid=w_out, exit_tid=exit_tid,
        in_axis=in_axis, gate_offset=gate_offset, out_in_axis=out_in_axis,
        norm_topk_prob=choice_plan.norm_topk_prob, choice_bias_tid=choice_plan.choice_bias_tid,
        extra_attrs={"routing_from_graph": "choice_score", "choice_routing": choice_plan.to_dict()},
    )
    new_order = [u for u in execution_order if u not in interior]
    pos = {u: i for i, u in enumerate(new_order)}
    dep_tids = [hidden_tid, choice_plan.logits_tid] + ([choice_plan.choice_bias_tid] if choice_plan.choice_bias_tid is not None else [])
    latest = max((pos[p] for p in (producer_map.get(t) for t in dep_tids) if p in pos), default=-1)
    earliest = min((pos[c] for c in consumer_map.get(exit_tid, []) if c in pos), default=len(new_order))
    if latest + 1 > earliest:
        return declined_cls("an input of the grouped_mm block is produced after its output is read")
    new_order.insert(latest + 1, fused_uid)
    ops[fused_uid] = fused_op
    for u in interior:
        ops.pop(u, None)
    execution_order[:] = new_order
    return (len(interior), fused_uid, fused_op, 0)


def fuse_choice_router_layer(
    dag, ops, execution_order, tensors, consumer_map, producer_map, topk_uid: str,
    *, choice_refusal, norm_topk_prob: bool, declared: bool, swept_whole: bool, tok,
    declined, refusals, declined_cls, fuse_gmm_fn, fuse_one_fn,
):
    """Parse and fuse a choice-score MoE router via grouped_mm or per-expert matcher."""
    plan = parse_choice_routing(ops, tensors, consumer_map, producer_map, topk_uid, declined_cls)
    if isinstance(plan, declined_cls):
        reason = f"{choice_refusal.reason} ({plan.reason})"
        declined[topk_uid] = reason
        if refusals is not None:
            refusals[topk_uid] = reason
        return None, swept_whole
    result = fuse_gmm_fn(
        dag, ops, execution_order, tensors, consumer_map, producer_map, topk_uid,
        choice_plan=plan, declared_norm=norm_topk_prob if declared else None, tok=tok,
    )
    if isinstance(result, declined_cls):
        gmm_reason = result.reason
        result = fuse_one_fn(
            dag, ops, execution_order, tensors, consumer_map, producer_map, topk_uid,
            norm_topk_prob=norm_topk_prob, choice_plan=plan,
            declared_norm=norm_topk_prob if declared else None,
            whole_sweep=not swept_whole, tok=tok,
        )
        swept_whole = swept_whole or result is not None
        if result is None:
            declined[topk_uid] = gmm_reason
            if refusals is not None:
                refusals[topk_uid] = gmm_reason
            if os.environ.get("NBX_DEBUG") or os.environ.get("NBX_MOE_FUSION_LOG"):
                _log.debug(f"[MoE Fusion] {topk_uid}: not fused — {gmm_reason}")
    return result, swept_whole


def resolve_one_layer_routing_tids(blend, choice_plan: Optional[ChoiceRoutingPlan], declared_norm, topk_uid: str, gate_scores_tid: Optional[str]) -> List[str]:
    """Validate optional choice_plan norm_topk_prob and return routing input tensor IDs for ``_fuse_one_moe_layer``."""
    if choice_plan is not None and declared_norm is not None and bool(declared_norm) != choice_plan.norm_topk_prob:
        raise RuntimeError(
            f"ZERO FALLBACK: [MoE Fusion] {topk_uid}: the registry declares "
            f"norm_topk_prob={bool(declared_norm)} but the container's graph computes "
            f"norm_topk_prob={choice_plan.norm_topk_prob} — fix the data at its source.")
    if blend is not None:
        return [blend.indices_tid, blend.weights_tid]
    if choice_plan is not None:
        return [gate_scores_tid, choice_plan.choice_bias_tid] if choice_plan.choice_bias_tid is not None else [gate_scores_tid]
    return [gate_scores_tid]


def stamp_one_layer_routing_attrs(fused_op: Dict[str, Any], blend, choice_plan: Optional[ChoiceRoutingPlan]) -> None:
    """Attach blend or choice-score routing metadata to a fused per-expert MoE op."""
    if blend is not None:
        fused_op["attributes"]["topk_indices_tid"] = blend.indices_tid
        fused_op["attributes"]["topk_weights_tid"] = blend.weights_tid
        fused_op["attributes"]["gate_group"] = list(blend.members)
    elif choice_plan is not None:
        fused_op["attributes"]["norm_topk_prob"] = choice_plan.norm_topk_prob
        fused_op["attributes"]["routing_from_graph"] = "choice_score"
        fused_op["attributes"]["choice_routing"] = choice_plan.to_dict()


def select_one_layer_routing_seeds(blend, choice_plan, topk_uid: str, topk_data: Dict[str, Any], moe_op_uids: Set[str]):
    """Return ``(gate_scores_tid, seed_weights_tid, seed_indices_tid)`` and update ``moe_op_uids``."""
    if blend is not None:
        return None, blend.weights_tid, blend.indices_tid
    if choice_plan is not None:
        moe_op_uids.update(choice_plan.routing_ops)
        return choice_plan.logits_tid, choice_plan.weights_tid, choice_plan.indices_tid
    gate_scores_tid = _in0(topk_data)
    if gate_scores_tid is None:
        raise RuntimeError(f"[MoE Fusion] Cannot find gate_scores input for {topk_uid}")
    moe_op_uids.add(topk_uid)
    return gate_scores_tid, topk_data["output_tensor_ids"][0], topk_data["output_tensor_ids"][1]
