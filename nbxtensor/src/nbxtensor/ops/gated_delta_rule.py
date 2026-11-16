"""The gated delta rule: a recurrence over a sequence whose state is a dk x dv float32 matrix per batch row and value
head, read and written one token at a time (the linear-attention layer's stateful operation).

Per batch row b, value head h, its key head h // (Hv / Hk), and token t (S a dk x dv matrix):

    S'_t = exp(g_t) * S_{t-1}                       the decay
    u_t  = beta_t * (v_t - S'_t^T k_t)              the correction
    S_t  = S'_t + k_t (x) u_t                       the rank-one write
    o_t  = S_t^T q_t                                the read, AFTER the write

q and k are L2-normalised (`rsqrt(sum(x * x) + eps)`) when the op asks it, and q is scaled by `scale`. Inputs arrive at
the model's dtype and UN-expanded (q, k with Hk heads, v with Hv): the head expansion, the norm, the scale and the
float32 arithmetic are inside the kernel, never an expanded float32 copy. The output is at v's dtype; the state is
float32 in and out and is never narrowed between steps.

ONE kernel, the serial scan: one program per (batch row, value head, dv tile). A column of S depends only on its own
column (u and o are read column by column), so the dk x BV float32 tile stays on chip across the length loop; the whole
128 x 128 state (64 KB) fits neither a V100 program nor Apple's 32 KB threadgroup memory. BV and the warps are the
certified configuration of the key (dk, dv and the operands' dtypes), never tuned at run time. The token step IS the
scan at one token: one kernel serves a prompt and a token. The length L is symbolic: it is the kernel's loop bound, never
its key, its grid or a chunk; the state goes in and comes back float32, so a prompt in two calls, or a prompt then one
token at a time, is the same recurrence as one call.

Its float64 reference is `nbxtensor.ops.oracles.fp64.delta_rule_oracle`; its launch is `launch_keys.
gated_delta_rule_launches`; its cost is `nbxtensor.ops._cost.gated_delta_rule`.
"""
from __future__ import annotations

import triton
import triton.language as tl

from nbxtensor.backend.runtime import set_device
from nbxtensor.dtype.nbx_dtype import NBXDtype, parse_dtype
from nbxtensor.ops.launch_keys import gated_delta_rule_launches  # noqa: F401  the op's launches, beside it
from nbxtensor.ops._configs import nbx_autotune
from nbxtensor.tensor import NBXTensor

#: The dtypes q, k, v and beta may arrive in (the model's); g and the state are float32 by the op's contract.
GATED_DELTA_RULE_INPUT_DTYPES = (NBXDtype.float16, NBXDtype.bfloat16, NBXDtype.float32)
#: The compute dtypes this release runs the recurrence in.
GATED_DELTA_RULE_COMPUTE_DTYPES = (NBXDtype.float32,)
#: The shortest sequence the op runs: one token (a call with no token has nothing to read or write).
GATED_DELTA_RULE_MIN_LENGTH = 1

# The dv tile and warps, candidates the certification measures per key; one stage (the loop streams q, k, v once).
_GATED_DELTA_RULE_CONFIGS = [triton.Config({'BV': bv}, num_warps=w, num_stages=1)
                             for bv in (16, 32, 64) for w in (1, 2, 4, 8)]


@nbx_autotune(configs=_GATED_DELTA_RULE_CONFIGS, key=['DK', 'DV'], cache_results=True)
@triton.jit
def gated_delta_rule_scan_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr,     # [B, T, Hk, DK] x2, [B, T, Hv, DV], [B, T, Hv] x2: own strides
    s_in_ptr,                                 # [B, Hv, DK, DV] float32 (read only when HAS_STATE)
    o_ptr,                                    # [B, T, Hv, DV] at v's dtype
    s_out_ptr,                                # [B, Hv, DK, DV] float32, contiguous
    T, DK, DV, GROUPS,                        # GROUPS = Hv // Hk: value head h reads key head h // GROUPS
    sq_b, sq_t, sq_h, sq_d,
    sk_b, sk_t, sk_h, sk_d,
    sv_b, sv_t, sv_h, sv_d,
    sg_b, sg_t, sg_h,
    sb_b, sb_t, sb_h,
    ss_b, ss_h, ss_k, ss_v,
    so_b, so_t, so_h, so_d,
    eps, scale,
    HAS_STATE: tl.constexpr,
    QK_L2NORM: tl.constexpr,
    BLOCK_K: tl.constexpr,                    # DK rounded up to a power of two: the whole key axis in one tile
    BV: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    vb = tl.program_id(2).to(tl.int64)
    kh = h // GROUPS

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = vb * BV + tl.arange(0, BV)
    mask_k = offs_k < DK
    mask_v = offs_v < DV
    mask_s = mask_k[:, None] & mask_v[None, :]

    if HAS_STATE:
        s = tl.load(s_in_ptr + b * ss_b + h * ss_h + offs_k[:, None] * ss_k + offs_v[None, :] * ss_v,
                    mask=mask_s, other=0.0).to(tl.float32)
    else:
        s = tl.zeros((BLOCK_K, BV), dtype=tl.float32)

    for t in range(0, T):                     # a 32-bit bound: the Metal lowering refuses a 64-bit scf.for bound
        tt = t.to(tl.int64)
        q = tl.load(q_ptr + b * sq_b + tt * sq_t + kh * sq_h + offs_k * sq_d, mask=mask_k, other=0.0).to(tl.float32)
        k = tl.load(k_ptr + b * sk_b + tt * sk_t + kh * sk_h + offs_k * sk_d, mask=mask_k, other=0.0).to(tl.float32)
        if QK_L2NORM:
            q = q / tl.sqrt(tl.sum(q * q, 0) + eps)
            k = k / tl.sqrt(tl.sum(k * k, 0) + eps)
        q = q * scale
        v = tl.load(v_ptr + b * sv_b + tt * sv_t + h * sv_h + offs_v * sv_d, mask=mask_v, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + b * sg_b + tt * sg_t + h * sg_h).to(tl.float32)
        beta = tl.load(beta_ptr + b * sb_b + tt * sb_t + h * sb_h).to(tl.float32)
        s = s * tl.exp(g)                                       # the decay
        u = beta * (v - tl.sum(s * k[:, None], 0))              # the correction, [BV]
        s = s + k[:, None] * u[None, :]                         # the rank-one write
        o = tl.sum(s * q[:, None], 0)                           # the read, after the write
        tl.store(o_ptr + b * so_b + tt * so_t + h * so_h + offs_v * so_d,
                 o.to(o_ptr.dtype.element_ty), mask=mask_v)

    tl.store(s_out_ptr + ((b * tl.num_programs(1) + h) * DK + offs_k[:, None]) * DV + offs_v[None, :],
             s, mask=mask_s)


def _shape_of(name: str, t, rank: int) -> tuple:
    shape = tuple(t.shape)
    if len(shape) != rank:
        raise ValueError(f"gated_delta_rule: {name} has shape {shape}; it must have {rank} dimensions")
    return shape


def gated_delta_rule_shapes(q, k, v, g, beta, state=None) -> tuple:
    """(B, L, Hk, Hv, dk, dv) of the op's operands, each operand's shape checked against the contract; a shape the op
    does not take is refused by name."""
    B, L, Hk, dk = _shape_of("q", q, 4)
    Hv, dv = _shape_of("v", v, 4)[2:]
    expected = {"k": (B, L, Hk, dk), "v": (B, L, Hv, dv), "g": (B, L, Hv), "beta": (B, L, Hv)}
    if state is not None:
        expected["state"] = (B, Hv, dk, dv)
    for name, t in (("k", k), ("v", v), ("g", g), ("beta", beta), ("state", state)):
        if name in expected and tuple(t.shape) != expected[name]:
            raise ValueError(f"gated_delta_rule: {name} has shape {tuple(t.shape)}; q {tuple(q.shape)} and v "
                             f"{tuple(v.shape)} ask {expected[name]}")
    if Hk == 0 or Hv % Hk:
        raise ValueError(f"gated_delta_rule: {Hv} value heads are not a whole multiple of {Hk} key heads; each key "
                         f"head serves Hv / Hk consecutive value heads")
    if L < GATED_DELTA_RULE_MIN_LENGTH:
        raise ValueError(f"gated_delta_rule: a sequence of length {L}: the delta rule runs over at least "
                         f"{GATED_DELTA_RULE_MIN_LENGTH} token; a call with no token has no output to write")
    return B, L, Hk, Hv, dk, dv


def gated_delta_rule_dtypes(q, k, v, g, beta, state, compute_dtype) -> NBXDtype:
    """The compute dtype, after each operand's DTYPE (an `NBXDtype`; `state` None when the op starts from zeros) is
    checked against the contract (q, k, v, beta at a model dtype; g and the state float32); a dtype the op does not
    take is refused by name. One check for the wrapper and its cost (`_cost.gated_delta_rule`): the declaration and the
    execution refuse the same operands."""
    try:
        cd = parse_dtype(compute_dtype)
    except KeyError:
        raise ValueError(f"gated_delta_rule: compute_dtype {compute_dtype!r} is not a dtype") from None
    if cd not in GATED_DELTA_RULE_COMPUTE_DTYPES:
        raise ValueError(f"gated_delta_rule: compute_dtype {cd.name} is refused: this release computes the recurrence "
                         f"in {[d.name for d in GATED_DELTA_RULE_COMPUTE_DTYPES]} only")
    for name, dt in (("q", q), ("k", k), ("v", v), ("beta", beta)):
        if dt not in GATED_DELTA_RULE_INPUT_DTYPES:
            raise ValueError(f"gated_delta_rule: {name} is {dt.name}; it must be one of "
                             f"{[d.name for d in GATED_DELTA_RULE_INPUT_DTYPES]}")
    for name, dt in (("g", g), ("state", state)):
        if dt is not None and dt != NBXDtype.float32:
            raise ValueError(f"gated_delta_rule: {name} is {dt.name}; it must be float32 (the op's contract: "
                             f"the decay and the state are float32, the state never narrowed)")
    return cd


def gated_delta_rule_wrapper(q, k, v, g, beta, state=None, *, qk_l2norm: bool, eps: float, scale: float,
                             compute_dtype):
    """The gated delta rule over q, k [B, L, Hk, dk], v [B, L, Hv, dv], g, beta [B, L, Hv] from `state` [B, Hv, dk, dv]
    float32 (None: zeros) -> (o [B, L, Hv, dv] at v's dtype, state_out [B, Hv, dk, dv] float32, a new tensor).

    `qk_l2norm` (with `eps`) and `scale` are the op's attributes, `compute_dtype` the dtype it computes in (float32
    only in this release); none has a default. L is any length from one token: the state handed back is the next
    call's start. The launch is the certified configuration of the key `launch_keys.gated_delta_rule_launches` forms
    (the length is not in it). Refused by name: a shape, a dtype or a length the op does not take.
    """
    B, L, Hk, Hv, dk, dv = gated_delta_rule_shapes(q, k, v, g, beta, state)
    gated_delta_rule_dtypes(*(None if t is None else t.nbx_dtype for t in (q, k, v, g, beta, state)), compute_dtype)
    out = NBXTensor.empty((B, L, Hv, dv), dtype=v.nbx_dtype, device=v.device)
    state_out = NBXTensor.empty((B, Hv, dk, dv), dtype=NBXDtype.float32, device=v.device)
    state_in = state_out if state is None else state
    set_device(v)
    gated_delta_rule_scan_kernel[lambda META: (B, Hv, triton.cdiv(dv, META['BV']))](
        q, k, v, g, beta, state_in, out, state_out,
        L, dk, dv, Hv // Hk,
        *q.stride(), *k.stride(), *v.stride(), *g.stride(), *beta.stride(), *state_in.stride(), *out.stride(),
        float(eps), float(scale),
        HAS_STATE=state is not None, QK_L2NORM=bool(qk_l2norm), BLOCK_K=triton.next_power_of_2(dk),
    )
    return out, state_out


gated_delta_rule_wrapper.self_manages_dtype = True

