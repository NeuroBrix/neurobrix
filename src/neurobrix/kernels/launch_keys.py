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


# ---------------------------------------------------------------------------------------------
# Convolutions: conv2d (and conv1d as H=1, conv3d as kt conv2d launches), depthwise, bands.
# ---------------------------------------------------------------------------------------------

_NARROWEST_FIRST = (F16, BF16, F32)


def conv_width_key(in_h: int, out_h: int, in_w: int, out_w: int, kw: int, stride_w: int, pad_w: int,
                   dil_w: int) -> Tuple[int, int]:
    """The width a convolution's key carries: a ONE-ROW convolution (a 1-D conv over a sequence)
    keys its width on the profile's W ladder, the output width derived from the input's top by the
    convolution's arithmetic; a 2-D convolution keeps its exact extents."""
    if int(in_h) == 1 and int(out_h) == 1:
        top = int(bucket_of("W", int(in_w)))
        return top, (top + 2 * int(pad_w) - int(dil_w) * (int(kw) - 1) - 1) // int(stride_w) + 1
    return int(in_w), int(out_w)


def conv_dtypes(x: NBXDtype, w: NBXDtype) -> Tuple[NBXDtype, NBXDtype]:
    """A convolution's operands, narrowed to the narrowest of the pair when they differ."""
    if x != w:
        n = next(d for d in _NARROWEST_FIRST if d in (x, w))
        return n, n
    return x, w


def conv_out_hw(in_h, in_w, kh, kw, sh, sw, ph, pw, dh, dw) -> Tuple[int, int]:
    return ((in_h + 2 * ph - dh * (kh - 1) - 1) // sh + 1,
            (in_w + 2 * pw - dw * (kw - 1) - 1) // sw + 1)


def conv2d_route(N: int, in_c: int, out_c: int, out_h: int, out_w: int, dh: int, dw: int, groups: int,
                 x: NBXDtype, band_bytes: int, depthwise_enabled: bool = True) -> str:
    """"depthwise" (groups == in_c == out_c at dilation 1), "band" (the output, sized at the
    input's dtype, over `band_bytes`) or "plain" — the order the wrapper tests them in."""
    from neurobrix.kernels.nbx_tensor import dtype_size
    if depthwise_enabled and groups == in_c and groups == out_c and dh == 1 and dw == 1:
        return "depthwise"
    if N * out_c * out_h * out_w * dtype_size(x) > band_bytes:
        return "band"
    return "plain"


def conv2d_launches(N: int, in_c: int, in_h: int, in_w: int, out_c: int, kh: int, kw: int,
                    sh: int, sw: int, ph: int, pw: int, dh: int, dw: int, groups: int,
                    x: NBXDtype, w: NBXDtype, compute: Optional[NBXDtype], band_bytes: int,
                    depthwise_enabled: bool = True) -> List[Launch]:
    """`conv2d_wrapper` on a 4-D weight: narrowing; the depthwise stencil when groups == in_c ==
    out_c at dilation 1; band streaming (per-band recursion at the original padding) when the
    output, sized at the INPUT's dtype, exceeds `band_bytes`; otherwise one `conv2d_forward_kernel`
    launch. The output dtype is the run's compute dtype when set (`_NBX_COMPUTE_DTYPE`), else the
    input's. REPRODUCED, not fixed: the plain kernel's `fp16` key flag compares a Triton dtype to
    an IntEnum and is always False (named for a decision — fixing it re-keys every conv entry)."""
    from neurobrix.kernels.nbx_tensor import dtype_size
    out_h, out_w = conv_out_hw(in_h, in_w, kh, kw, sh, sw, ph, pw, dh, dw)
    if out_h <= 0 or out_w <= 0:
        return []
    x, w = conv_dtypes(x, w)
    out = compute if compute is not None else x
    route = conv2d_route(N, in_c, out_c, out_h, out_w, dh, dw, groups, x, band_bytes, depthwise_enabled)
    if route == "depthwise":
        iw_k, ow_k = conv_width_key(in_h, out_h, in_w, out_w, kw, sw, pw, 1)
        return [(DEPTHWISE, (in_c, in_h, iw_k, out_h, ow_k, kh, kw, sh, sw, ph, pw, x == F16,
                             tag(x), tag(w), tag(out)))]
    xb = dtype_size(x)
    if route == "band":
        band_target = max(1, band_bytes // 2)
        rows_per_band = max(1, band_target // max(1, N * out_c * out_w * xb))
        tile_factor = max(1, (out_h + rows_per_band - 1) // rows_per_band)
        band_oh = (out_h + tile_factor - 1) // tile_factor
        out_launches: List[Launch] = []
        for oh0 in range(0, out_h, band_oh):
            oh1 = min(oh0 + band_oh, out_h)
            ih0 = max(0, oh0 * sh - ph)
            ih1 = min(in_h, (oh1 - 1) * sh + dh * (kh - 1) + 1 - ph)
            if ih1 <= ih0:
                continue
            for l in conv2d_launches(N, in_c, ih1 - ih0, in_w, out_c, kh, kw, sh, sw, ph, pw, dh, dw,
                                     groups, x, w, compute, band_bytes, depthwise_enabled):
                if l not in out_launches:
                    out_launches.append(l)
        return out_launches
    iw_k, ow_k = conv_width_key(in_h, out_h, in_w, out_w, kw, sw, pw, dw)
    return [(CONV2D, (N, in_c, in_h, iw_k, out_c, out_h, ow_k, kh, kw, sh, sw, ph, pw, dh, dw, groups,
                      False, tag(x), tag(w), tag(out)))]


def conv_launches(x_shape, w_shape, stride, padding, dilation, transposed: bool, groups: int,
                  x: NBXDtype, w: NBXDtype, compute: Optional[NBXDtype], band_bytes: int,
                  depthwise_enabled: bool = True) -> List[Launch]:
    """`conv2d_wrapper` by the weight's rank: a transposed convolution forms no autotuned key; a
    3-D weight is conv1d as conv2d at H = 1; a 5-D weight is conv3d as kt conv2d launches over
    [B*T_out, Cin, H, W] (identical keys). REPRODUCED as the census forms it: the chunked conv3d
    variant is gated by the driver's free bytes, which a census answers -1 — a live run can chunk
    and form keys this derivation does not (named)."""
    def _n(v, n):
        v = list(v) if isinstance(v, (list, tuple)) else [v]
        return v + [v[-1]] * (n - len(v)) if len(v) < n else v[:n]
    if transposed:
        return []
    if len(w_shape) == 3:
        N, C, L = x_shape
        Co, Cg, K = w_shape
        s, p, d = _n(stride, 1)[0], _n(padding, 1)[0], _n(dilation, 1)[0]
        return conv2d_launches(N, C, 1, L, Co, 1, K, 1, s, 0, p, 1, d, groups, x, w, compute,
                               band_bytes, depthwise_enabled)
    if len(w_shape) == 5:
        B, Cin, T, H, W = x_shape
        Cout, Cg, kt, kh, kw = w_shape
        st, sh, sw = _n(stride, 3)
        pt, ph, pw = _n(padding, 3)
        dt, dh, dw = _n(dilation, 3)
        T_out = (T + 2 * pt - dt * (kt - 1) - 1) // st + 1
        if T_out <= 0:
            return []
        return conv2d_launches(B * T_out, Cin, H, W, Cout, kh, kw, sh, sw, ph, pw, dh, dw, groups,
                               x, w, compute, band_bytes, depthwise_enabled)
    N, C, H, W = x_shape
    Co, Cg, kh, kw = w_shape
    sh, sw = _n(stride, 2)
    ph, pw = _n(padding, 2)
    dh, dw = _n(dilation, 2)
    return conv2d_launches(N, C, H, W, Co, kh, kw, sh, sw, ph, pw, dh, dw, groups, x, w, compute,
                           band_bytes, depthwise_enabled)
