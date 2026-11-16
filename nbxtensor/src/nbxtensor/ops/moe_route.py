"""The routing of a mixture of experts: which top_k of E experts a token goes to, and with which float32 weights (the
router's whole skeleton, from the hidden row to the routing a dispatch takes as precomputed).

Per token row x [H], with the router weight W [E, H]:

    logits = fp32(x) . fp32(W)^T                          the router product, float32 operands and accumulator
    s      = sigmoid(logits) | softmax(logits)            the scores (`score_fn`)
    c      = s + choice_bias                              the choice values (the bias steers the CHOICE only)
    groups = the E experts cut into n_group consecutive groups of E / n_group
    gscore = per group: the sum of its two largest c (`top2_sum`) or its largest c (`max`)
    c      = c in the topk_group best groups, `fill` elsewhere
    idx    = the top_k largest c
    w      = s[idx]                                       the UN-biased scores of the chosen experts
    w      = w / (sum(w) + renorm_eps)                    when `renorm_eps` is not None
    w      = w * scale

n_group = 1 keeps every expert, so it is exactly the ungrouped routing. The slots come in descending choice order, the
lower expert id first on an exact tie; the order of the slots is NOT part of the contract (a set per token, weights
paired by expert id). Outputs: indices [T, top_k] int64, weights [T, top_k] float32.

ONE kernel: one program per token row. The router product runs over H in BLOCK_H tiles into an E-long float32
accumulator, then the scores, the bias, the group reduction, the two selections, the gather, the renormalisation and
the scale stay in registers: E <= `MOE_ROUTE_MAX_EXPERTS` experts fit one row per program. BLOCK_H and the warps are
the certified configuration of the key (E, H, top_k and the operands' dtypes), never tuned at run time. n_group is
not in the key: the grouped and the ungrouped routing of one geometry share one configuration, so n_group 1 and every
group kept give the same bytes. Every
loop bound is 32-bit (the Metal lowering refuses a 64-bit scf.for bound). This release runs the token step
(`MOE_ROUTE_TOKENS`); the routing over a symbolic number of tokens is the next lot.

Its float64 reference is `nbxtensor.ops.oracles.fp64.moe_route_oracle`; its launch is `launch_keys.moe_route_launches`.
"""
from __future__ import annotations

import math

import triton
import triton.language as tl

from nbxtensor.backend.runtime import set_device
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.ops.launch_keys import moe_route_launches  # noqa: F401  the op's launches, beside it
from nbxtensor.ops._configs import nbx_autotune
from nbxtensor.tensor import NBXTensor

#: The dtypes the hidden row and the router weight may arrive in; the choice bias is float32 by the op's contract.
MOE_ROUTE_INPUT_DTYPES = (NBXDtype.float16, NBXDtype.bfloat16, NBXDtype.float32)
#: The score functions and the group scores the op computes.
MOE_ROUTE_SCORE_FNS = ("sigmoid", "softmax")
MOE_ROUTE_GROUP_SCORES = ("top2_sum", "max")
#: The most experts one program holds in registers (one row per program).
MOE_ROUTE_MAX_EXPERTS = 256
#: The token counts this release runs: the token step (the routing over a symbolic count is the next lot).
MOE_ROUTE_TOKENS = (1,)

# The hidden tile and warps, candidates the certification measures per key; one stage (the loop streams W once).
_MOE_ROUTE_CONFIGS = [triton.Config({'BLOCK_H': bh}, num_warps=w, num_stages=1)
                      for bh in (16, 32, 64, 128) for w in (2, 4, 8)]


@nbx_autotune(configs=_MOE_ROUTE_CONFIGS, key=['E', 'H', 'TOP_K'], cache_results=True)
@triton.jit
def moe_route_kernel(
    x_ptr,                                    # [T, H] at the model's dtype, own strides
    w_ptr,                                    # [E, H] the router weight, own strides
    bias_ptr,                                 # [E] float32, own stride (read only when HAS_BIAS)
    idx_ptr,                                  # [T, TOP_K] int64, contiguous
    wt_ptr,                                   # [T, TOP_K] float32, contiguous
    E, H, TOP_K, N_GROUP, TOPK_GROUP,
    sx_t, sx_h, sw_e, sw_h, sb_e,
    fill, renorm_eps, scale,
    SIGMOID: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUPED: tl.constexpr,                    # N_GROUP > 1
    GROUP_TOP2: tl.constexpr,                 # group score = sum of the two largest (else the largest)
    RENORM: tl.constexpr,
    BLOCK_E: tl.constexpr,                    # E rounded up to a power of two
    BLOCK_G: tl.constexpr,                    # N_GROUP rounded up to a power of two
    BLOCK_K: tl.constexpr,                    # TOP_K rounded up to a power of two
    BLOCK_H: tl.constexpr,
):
    NEG = float("-inf")
    t = tl.program_id(0).to(tl.int64)
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E

    acc = tl.zeros((BLOCK_E,), dtype=tl.float32)
    for h0 in range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        x = tl.load(x_ptr + t * sx_t + offs_h.to(tl.int64) * sx_h, mask=mask_h, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs_e[:, None].to(tl.int64) * sw_e + offs_h[None, :].to(tl.int64) * sw_h,
                    mask=mask_e[:, None] & mask_h[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], 1)

    if SIGMOID:
        s = 1.0 / (1.0 + tl.exp(-acc))
    else:
        lg = tl.where(mask_e, acc, NEG)
        p = tl.exp(lg - tl.max(lg, 0))
        s = p / tl.sum(p, 0)
    s = tl.where(mask_e, s, 0.0)
    c = s
    if HAS_BIAS:
        c = s + tl.load(bias_ptr + offs_e.to(tl.int64) * sb_e, mask=mask_e, other=0.0)
    c = tl.where(mask_e, c, NEG)

    if GROUPED:
        grp = offs_e // (E // N_GROUP)
        offs_g = tl.arange(0, BLOCK_G)
        gs = tl.full((BLOCK_G,), NEG, tl.float32)
        for g in range(0, N_GROUP):
            v = tl.where(grp == g, c, NEG)
            m1 = tl.max(v, 0)
            if GROUP_TOP2:
                i1 = tl.min(tl.where(v == m1, offs_e, BLOCK_E), 0)
                m1 = m1 + tl.max(tl.where(offs_e == i1, NEG, v), 0)
            gs = tl.where(offs_g == g, m1, gs)
        kept_g = tl.zeros((BLOCK_G,), dtype=tl.int32)
        keep = tl.zeros((BLOCK_E,), dtype=tl.int32)
        for j in range(0, TOPK_GROUP):
            avail_g = (kept_g == 0) & (offs_g < N_GROUP)
            mg = tl.max(tl.where(avail_g, gs, NEG), 0)
            gi = tl.min(tl.where(avail_g & ((gs == mg) | (gs != gs)), offs_g, BLOCK_G), 0)
            kept_g = tl.where(offs_g == gi, 1, kept_g)
            keep = tl.where(grp == gi, 1, keep)
        c = tl.where(mask_e, tl.where(keep != 0, c, fill), NEG)

    offs_k = tl.arange(0, BLOCK_K)
    taken = tl.zeros((BLOCK_E,), dtype=tl.int32)
    idx = tl.zeros((BLOCK_K,), dtype=tl.int32)
    wk = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for j in range(0, TOP_K):
        avail = (taken == 0) & mask_e
        m = tl.max(tl.where(avail, c, NEG), 0)
        i = tl.min(tl.where(avail & ((c == m) | (c != c)), offs_e, BLOCK_E), 0)   # a NaN lane is chosen, never none
        taken = tl.where(offs_e == i, 1, taken)
        idx = tl.where(offs_k == j, i, idx)
        wk = tl.where(offs_k == j, tl.sum(tl.where(offs_e == i, s, 0.0), 0), wk)
    if RENORM:
        wk = wk / (tl.sum(wk, 0) + renorm_eps)
    wk = wk * scale
    mask_k = offs_k < TOP_K
    tl.store(idx_ptr + t * TOP_K + offs_k, idx.to(tl.int64), mask=mask_k)
    tl.store(wt_ptr + t * TOP_K + offs_k, wk, mask=mask_k)


def _shape_of(name: str, t, rank: int) -> tuple:
    shape = tuple(t.shape)
    if len(shape) != rank:
        raise ValueError(f"moe_route: {name} has shape {shape}; it must have {rank} dimension(s)")
    return shape


def moe_route_shapes(hidden, router_weight, choice_bias) -> tuple:
    """(T, H, E) of the op's operands, each shape checked against the contract; a shape the op does not take, a token
    count this release does not run or more experts than one program holds are refused by name."""
    T, H = _shape_of("hidden", hidden, 2)
    E, Hw = _shape_of("router_weight", router_weight, 2)
    if Hw != H:
        raise ValueError(f"moe_route: router_weight has shape {(E, Hw)}; hidden {(T, H)} asks [E, {H}]")
    if choice_bias is not None and tuple(choice_bias.shape) != (E,):
        raise ValueError(f"moe_route: choice_bias has shape {tuple(choice_bias.shape)}; router_weight {(E, H)} "
                         f"asks ({E},)")
    if not 1 <= E <= MOE_ROUTE_MAX_EXPERTS:
        raise ValueError(f"moe_route: {E} experts: one program holds 1 to {MOE_ROUTE_MAX_EXPERTS} experts")
    if T not in MOE_ROUTE_TOKENS:
        raise ValueError(f"moe_route: {T} tokens: this release routes the token counts {MOE_ROUTE_TOKENS} only (the "
                         f"token step); the routing over more tokens is not proven yet")
    return T, H, E


def moe_route_dtypes(hidden, router_weight, choice_bias) -> None:
    """Each operand's dtype checked against the contract (hidden and router weight at a model dtype, the choice bias
    float32); a dtype the op does not take is refused by name."""
    for name, t in (("hidden", hidden), ("router_weight", router_weight)):
        if t.nbx_dtype not in MOE_ROUTE_INPUT_DTYPES:
            raise ValueError(f"moe_route: {name} is {t.nbx_dtype.name}; it must be one of "
                             f"{[d.name for d in MOE_ROUTE_INPUT_DTYPES]}")
    if choice_bias is not None and choice_bias.nbx_dtype != NBXDtype.float32:
        raise ValueError(f"moe_route: choice_bias is {choice_bias.nbx_dtype.name}; it must be float32 (the op's "
                         f"contract: the bias is kept float32 as stored, never narrowed)")


def _check_selection(E: int, n_group: int, topk_group: int, group_score: str, top_k: int) -> None:
    if n_group < 1 or E % n_group:
        raise ValueError(f"moe_route: n_group {n_group} does not cut {E} experts into equal groups")
    if group_score not in MOE_ROUTE_GROUP_SCORES:
        raise ValueError(f"moe_route: group_score {group_score!r} is not one of {MOE_ROUTE_GROUP_SCORES}")
    if group_score == "top2_sum" and n_group > 1 and E // n_group < 2:
        raise ValueError(f"moe_route: group_score 'top2_sum' over groups of {E // n_group} expert: a group must hold "
                         f"at least two")
    if not 1 <= topk_group <= n_group:
        raise ValueError(f"moe_route: topk_group {topk_group} is not between 1 and n_group {n_group}")
    kept = topk_group * (E // n_group)
    if not 1 <= top_k <= kept:
        raise ValueError(f"moe_route: top_k {top_k} is not between 1 and the {kept} experts of the {topk_group} kept "
                         f"group(s)")


def moe_route_attributes(E: int, *, score_fn, top_k, n_group, topk_group, group_score, fill, renorm_eps,
                         scale) -> None:
    """The op's attributes checked against each other and against E; an attribute the op does not take is refused by
    name."""
    if score_fn not in MOE_ROUTE_SCORE_FNS:
        raise ValueError(f"moe_route: score_fn {score_fn!r} is not one of {MOE_ROUTE_SCORE_FNS}")
    _check_selection(E, int(n_group), int(topk_group), group_score, int(top_k))
    if math.isnan(fill) or fill == math.inf:
        raise ValueError(f"moe_route: fill {fill} would rank a dropped group's experts first or never compare; it "
                         f"must be a number below +inf (0.0 or -inf in the recorded routers)")
    if renorm_eps is not None and not (math.isfinite(renorm_eps) and renorm_eps >= 0):
        raise ValueError(f"moe_route: renorm_eps {renorm_eps} must be None (no renormalisation) or a finite number "
                         f">= 0")
    if not math.isfinite(scale):
        raise ValueError(f"moe_route: scale {scale} is not a finite number")


def moe_route(hidden, router_weight, *, choice_bias, score_fn, top_k, n_group, topk_group, group_score, fill,
              renorm_eps, scale):
    """The routing of the token rows `hidden` [T, H] over the experts of `router_weight` [E, H], with `choice_bias`
    [E] float32 (or None) -> (indices [T, top_k] int64, weights [T, top_k] float32), new tensors on hidden's device.

    `score_fn` ('sigmoid' | 'softmax'), `n_group`, `topk_group`, `group_score` ('top2_sum' | 'max'), `fill` (the
    value a dropped group's experts take before the final choice), `top_k`, `renorm_eps` (None: no renormalisation)
    and `scale` are the op's attributes; none has a default. The router product and everything after it are float32
    by the op's contract. The launch is the certified configuration of the key `launch_keys.moe_route_launches`
    forms. Refused by name: a shape, a dtype, a token count or an attribute the op does not take.
    """
    T, H, E = moe_route_shapes(hidden, router_weight, choice_bias)
    moe_route_dtypes(hidden, router_weight, choice_bias)
    moe_route_attributes(E, score_fn=score_fn, top_k=top_k, n_group=n_group, topk_group=topk_group,
                         group_score=group_score, fill=float(fill), renorm_eps=renorm_eps, scale=float(scale))
    top_k, n_group = int(top_k), int(n_group)
    indices = NBXTensor.empty((T, top_k), dtype=NBXDtype.int64, device=hidden.device)
    weights = NBXTensor.empty((T, top_k), dtype=NBXDtype.float32, device=hidden.device)
    set_device(hidden)
    moe_route_kernel[(T,)](
        hidden, router_weight, weights if choice_bias is None else choice_bias, indices, weights,
        E, H, top_k, n_group, int(topk_group),
        *hidden.stride(), *router_weight.stride(), 0 if choice_bias is None else choice_bias.stride()[0],
        float(fill), 0.0 if renorm_eps is None else float(renorm_eps), float(scale),
        SIGMOID=score_fn == "sigmoid", HAS_BIAS=choice_bias is not None, GROUPED=n_group > 1,
        GROUP_TOP2=group_score == "top2_sum", RENORM=renorm_eps is not None,
        BLOCK_E=triton.next_power_of_2(E), BLOCK_G=triton.next_power_of_2(n_group),
        BLOCK_K=triton.next_power_of_2(top_k),
    )
    return indices, weights


moe_route.self_manages_dtype = True
