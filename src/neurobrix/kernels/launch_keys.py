"""The autotuned kernels' launch decisions as PURE functions of shapes and dtypes.

The owner's ruling (2026-09-29 01:37): the census is a DERIVATION from the graphs, never an
execution. Every key a run forms is a function of the op's operand shapes, operand dtypes and
attributes, the hardware's `has_native_bf16`, and the profile's bucket ladders — so the key is
computed here, from those alone, and the wrappers make their launch decisions through the SAME
functions: one implementation, which the derived census calls without launching anything and the
wrappers call before they launch. A census shadow run is the proof that the two agree
(`census.expect_launches`: every recorded key must be one the op's derivation named).

Nothing here touches a tensor or a device. The dtypes are `NBXDtype` members; a key's dtype tags
are the launcher's own spelling (`str` of the Triton dtype, via `_get_tl_dtype`), in the kernel's
argument order, exactly as `autotune_cache.key_of` appends them.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from neurobrix.kernels.autotune_bucket import bucket_of
from neurobrix.kernels.nbx_tensor import NBXDtype, _get_tl_dtype

MATMUL = "neurobrix.kernels.ops.matmul.matmul_kernel"
ADDMM = "neurobrix.kernels.ops.matmul.addmm_kernel"
BADDBMM = "neurobrix.kernels.ops.baddbmm_op.baddbmm_kernel"
CONV2D = "neurobrix.kernels.ops.conv2d.conv2d_forward_kernel"
DEPTHWISE = "neurobrix.kernels.ops.depthwise_conv2d.depthwise_conv2d_kernel"

F32, BF16, F16 = NBXDtype.float32, NBXDtype.bfloat16, NBXDtype.float16
_HALF = (F16, BF16)
_WIDEST_FIRST = (F32, BF16, F16)

Launch = Tuple[str, tuple]          # (kernel qualified name, key tuple as key_of forms it)


def tag(dt: NBXDtype) -> str:
    """A dtype as the autotune key spells it (`str` of the Triton dtype the tensor reports)."""
    return str(_get_tl_dtype(dt))


def _widest(a: NBXDtype, b: NBXDtype) -> NBXDtype:
    return next(d for d in _WIDEST_FIRST if d in (a, b))


def matmul_out_dtype(dt: NBXDtype, M: int, force_fp32: bool, native_bf16: bool,
                     force_accum: bool = False) -> NBXDtype:
    """The store dtype of mm/bmm/addmm/mv/addmv (see `wrappers._matmul_out_dtype` for the why):
    fp32 for any half input under NBX_FORCE_FP32_ACCUM; fp32 for fp16 on hardware without native
    bf16; fp32 for a half input when M <= 4 or when forced; otherwise the input dtype."""
    half = dt in _HALF
    if force_accum and half:
        return F32
    if dt == F16 and not native_bf16:
        return F32
    if half and (M <= 4 or force_fp32):
        return F32
    return dt


def mm_dtypes(a: NBXDtype, b: NBXDtype, native_bf16: bool, force_accum: bool = False):
    """`mm`'s operand decisions: (a at launch, b at launch, PROMOTE_A, PROMOTE_B). An fp16
    activation on hardware without native bf16 is widened IN the kernel (PROMOTE_A, a stays fp16
    in memory); a narrow weight under an fp32 activation is widened in the kernel (PROMOTE_B, any
    hardware); any other mismatch widens both in memory to the widest of the pair."""
    if force_accum and a in _HALF:
        a = F32
    if force_accum and b in _HALF:
        b = F32
    promote_a = (not native_bf16) and a == F16
    a_eff = F32 if promote_a else a
    promote_b = a_eff == F32 and b in _HALF
    if a_eff != b and not promote_b:
        if promote_a:
            a, promote_a = F32, False
        w = _widest(a, b)
        a, b = w, w
    return a, b, promote_a, promote_b


def addmm_dtypes(a: NBXDtype, b: NBXDtype, bias: NBXDtype, native_bf16: bool,
                 force_accum: bool = False):
    """`addmm`'s operand decisions: (a, b, bias at launch, PROMOTE_A, PROMOTE_B, PROMOTE_BIAS).
    As `mm`, except PROMOTE_B is gated on hardware without native bf16 and on an fp16 weight
    only; the bias is widened in the kernel under an fp32 activation, else cast to a's dtype."""
    if force_accum and a in _HALF:
        a = F32
    if force_accum and b in _HALF:
        b = F32
    promote_a = (not native_bf16) and a == F16
    a_eff = F32 if promote_a else a
    promote_b = (not native_bf16) and a_eff == F32 and b == F16
    if a_eff != b and not promote_b:
        if promote_a:
            a, promote_a = F32, False
        w = _widest(a, b)
        a, b = w, w
        a_eff = a
    promote_bias = bias != a_eff and a_eff == F32 and bias in _HALF
    if bias != a_eff and not promote_bias:
        bias = a_eff
    return a, b, bias, promote_a, promote_b, promote_bias


def bmm_dtypes(a: NBXDtype, b: NBXDtype, native_bf16: bool, force_accum: bool = False):
    """`bmm`'s operand decisions: (a, b at launch, PROMOTE_B). An fp16 activation on hardware
    without native bf16 is widened in memory (bmm has no PROMOTE_A); an fp16 weight under fp32 is
    widened in the kernel on that hardware; any other mismatch widens both to the widest."""
    if force_accum and a in _HALF:
        a = F32
    if force_accum and b in _HALF:
        b = F32
    if (not native_bf16) and a == F16:
        a = F32
    promote_b = (not native_bf16) and a == F32 and b == F16
    if a != b and not promote_b:
        w = _widest(a, b)
        a, b = w, w
    return a, b, promote_b


def mm_launches(M: int, K: int, N: int, a: NBXDtype, b: NBXDtype, native_bf16: bool,
                force_accum: bool = False) -> List[Launch]:
    """`mm` (and `mm_epilogue`): M <= 4 runs per-row GEMV (no autotuned kernel); otherwise one
    key — row bands above 2^31 output elements keep the whole shape's M bucket."""
    if M <= 4:
        return []
    a, b, promote_a, promote_b = mm_dtypes(a, b, native_bf16, force_accum)
    out = matmul_out_dtype(a, M, promote_a, native_bf16, force_accum)
    ieee = (not native_bf16) and out == F32
    return [(MATMUL, (bucket_of("M", M), N, K, ieee, promote_b, tag(a), tag(b), tag(out)))]


def addmm_launches(M: int, K: int, N: int, a: NBXDtype, b: NBXDtype, bias: NBXDtype,
                   native_bf16: bool, force_accum: bool = False) -> List[Launch]:
    """`addmm` on a 2-D activation (an N-D one is flattened to (numel/K, K) first): M <= 4 runs
    per-row addmv; otherwise one key."""
    a, b, bias, promote_a, promote_b, _ = addmm_dtypes(a, b, bias, native_bf16, force_accum)
    if M <= 4:
        return []
    out = matmul_out_dtype(a, M, promote_a, native_bf16, force_accum)
    ieee = (not native_bf16) and out == F32
    return [(ADDMM, (bucket_of("M", M), N, K, ieee, promote_b, tag(a), tag(b), tag(bias), tag(out)))]


def bmm_launches(M: int, K: int, N: int, a: NBXDtype, b: NBXDtype, native_bf16: bool,
                 force_accum: bool = False) -> List[Launch]:
    """`bmm`: always one batched launch through `baddbmm_kernel` (HAS_BIAS False, the output
    passed as the bias pointer), every dim bucketed, the output fp32 for any half input."""
    a, b, promote_b = bmm_dtypes(a, b, native_bf16, force_accum)
    out = matmul_out_dtype(a, M, True, native_bf16, force_accum)
    ieee = (not native_bf16) and out == F32
    return [(BADDBMM, (bucket_of("M", M), bucket_of("N", N), bucket_of("K", K),
                       ieee, promote_b, False, tag(a), tag(b), tag(out), tag(out)))]


# ---------------------------------------------------------------------------------------------
# Attention (scaled_dot_product_attention): the route, then the math route's two bmm launches.
# ---------------------------------------------------------------------------------------------

def _pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def sdpa_chunk_rows(bound: int, batch: int, nheads: int, Tq: int, Tk: int,
                    min_chunk_rows: int, max_chunks: int) -> int:
    """Query rows per chunk that keep each chunk's fp32 scores within `bound`, aligned to the
    arch's row block (`memory.sdpa_math_min_chunk_rows`); 0 when the shape cannot chunk within
    the arch's chunk ceiling (`memory.sdpa_math_max_chunks`)."""
    if not min_chunk_rows:
        return 0
    rows = (bound // (batch * nheads * Tk * 4)) // min_chunk_rows * min_chunk_rows
    if rows >= min_chunk_rows and -(-Tq // rows) <= max_chunks:
        return rows
    return 0


def sdpa_route(batch: int, nheads: int, Tq: int, Tk: int, D: int, Dv: int, budget_bytes: int,
               min_chunk_rows: int, max_chunks: int, force_math: bool = False) -> Tuple[str, int]:
    """The attention route: ("math", 0), ("chunked", rows) or ("flash", 0). Math when forced or
    when the value head dim differs; else by the fp32 scores' size against the executing device's
    budget — a non-power-of-two head dim falls back to a 2 GiB bound when the arch declares none;
    over the bound, chunked when the rows fit the arch's ceiling; otherwise flash."""
    if force_math or Dv != D:
        return ("math", 0)
    scores = batch * nheads * Tq * Tk * 4
    bound = budget_bytes if _pow2(D) else (budget_bytes or (2 << 30))
    if scores <= bound:
        return ("math", 0)
    if bound:
        rows = sdpa_chunk_rows(bound, batch, nheads, Tq, Tk, min_chunk_rows, max_chunks)
        if rows:
            return ("chunked", rows)
    return ("flash", 0)


def math_attention_launches(B: int, H: int, Hk: int, Tq: int, Tk: int, D: int, Dv: int,
                            q: NBXDtype, k: NBXDtype, v: NBXDtype, native_bf16: bool,
                            force_accum: bool = False) -> List[Launch]:
    """`_math_attention`: scores = bmm(q as [B*Hk, groups*Tq, D], k^T) — q in its (possibly
    rounded) dtype, k in its own; then p = softmax(scores) cast to v's dtype and bmm(p, v)."""
    M = (H // Hk) * Tq
    return (bmm_launches(M, D, Tk, q, k, native_bf16, force_accum)
            + bmm_launches(M, Tk, Dv, v, v, native_bf16, force_accum))


def chunked_math_attention_launches(B: int, H: int, Hk: int, Tq: int, Tk: int, D: int, Dv: int,
                                    q: NBXDtype, k: NBXDtype, v: NBXDtype, native_bf16: bool,
                                    rows: int, force_accum: bool = False) -> List[Launch]:
    """`_math_attention_chunked`: the math route over query-row chunks of `rows`, the last one the
    remainder — each chunk forms its own pair of keys."""
    out: List[Launch] = []
    for c in {min(rows, Tq)} | ({Tq % rows} if Tq % rows else set()):
        out += math_attention_launches(B, H, Hk, c, Tk, D, Dv, q, k, v, native_bf16, force_accum)
    return out
