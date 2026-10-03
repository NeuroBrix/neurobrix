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


def mm_out(M: int, a: NBXDtype, b: NBXDtype, native_bf16: bool, force_accum: bool = False) -> NBXDtype:
    """The dtype `mm` stores for these operands (its launch or its per-row GEMV alike)."""
    a, b, promote_a, _ = mm_dtypes(a, b, native_bf16, force_accum)
    return matmul_out_dtype(a, M, promote_a, native_bf16, force_accum)


def bmm_out(M: int, a: NBXDtype, b: NBXDtype, native_bf16: bool, force_accum: bool = False) -> NBXDtype:
    """The dtype `bmm` stores for these operands."""
    a, b, _ = bmm_dtypes(a, b, native_bf16, force_accum)
    return matmul_out_dtype(a, M, True, native_bf16, force_accum)


def matmul_route(a_rank: int, b_rank: int) -> str:
    """`matmul_wrapper`'s route by operand ranks (torch.matmul semantics): "mm" (2-D x 2-D), "bmm"
    (3-D x 3-D), "mv" (2-D x 1-D, the SIMT GEMV — no autotuned kernel), "batched" (N-D x 2-D:
    the leading dims folded into bmm's batch), "general" (N-D x N-D: both batches collapsed)."""
    if a_rank == 2 and b_rank == 2:
        return "mm"
    if a_rank == 3 and b_rank == 3:
        return "bmm"
    if a_rank == 2 and b_rank == 1:
        return "mv"
    if a_rank >= 3 and b_rank == 2:
        return "batched"
    if a_rank >= 3 and b_rank >= 3:
        return "general"
    raise ValueError(f"matmul: unsupported ranks {a_rank} x {b_rank}")


def matmul_launches(a_shape, b_shape, a: NBXDtype, b: NBXDtype, native_bf16: bool,
                    force_accum: bool = False) -> List[Launch]:
    """`matmul_wrapper` (and `linear_wrapper`, which calls it with the transposed weight): the
    launch of the route `matmul_route` names — bmm's key carries no batch."""
    route = matmul_route(len(a_shape), len(b_shape))
    if route == "mm":
        return mm_launches(a_shape[0], a_shape[1], b_shape[1], a, b, native_bf16, force_accum)
    if route == "mv":
        return []
    return bmm_launches(a_shape[-2], a_shape[-1], b_shape[-1], a, b, native_bf16, force_accum)


# ---------------------------------------------------------------------------------------------
# Recurrence and spectra: `lstm_wrapper` and the DFT route of the FFT wrappers, as the matmul
# launches their generic calls make.
# ---------------------------------------------------------------------------------------------

def lstm_launches(B: int, T: int, I: int, H: int, num_layers: int, bidirectional: bool,
                  x: NBXDtype, native_bf16: bool) -> List[Launch]:
    """`lstm_wrapper` -> `_lstm_run_direction`, per layer and direction: the input projection
    x[B,T,I] @ W_ih^T (matmul's N-D x 2-D route: `bmm` over B batches of [T, I]), then one
    h[B,H] @ W_hh^T `mm` per step. The weights and the initial state are cast to the input's
    dtype `x` (`cdt`, read once); the elementwise gate arithmetic promotes to the wider operand,
    so the state after a step is the widest of the projection's, the step mm's and `x`, and a
    later layer reads the concatenation at its FIRST operand's dtype (`NBXTensor.cat`): the
    forward direction's first step. The state's dtype is iterated to its fixed point (it can
    only widen), so the step keys are every dtype the state takes."""
    nd = 2 if bidirectional else 1
    out: List[Launch] = []
    layer_in = x
    for layer in range(int(num_layers)):
        first_fwd = None
        for d in range(nd):
            out += bmm_launches(T, I if layer == 0 else H * nd, 4 * H, layer_in, x, native_bf16)
            wx = bmm_out(T, layer_in, x, native_bf16)
            h, seen, first = x, set(), None
            for _step in range(max(int(T), 0)):
                if h in seen:
                    break
                seen.add(h)
                out += mm_launches(B, H, 4 * H, h, x, native_bf16)
                h = _widest(_widest(wx, mm_out(B, h, x, native_bf16)), x)
                first = h if first is None else first
            if d == 0:
                first_fwd = first if first is not None else x
        layer_in = first_fwd
    return list(dict.fromkeys(out))


def dft_r2c_launches(M: int, N: int, onesided: bool, native_bf16: bool) -> List[Launch]:
    """`fft_r2c_wrapper` on a length that is not a power of two: `_dft_r2c`, the frames flattened
    to [M, N] and widened to fp32, times the fp32 cos and -sin matrices (two `mm`). A power of
    two runs the radix-2 butterfly (no autotuned kernel)."""
    if N > 1 and (N & (N - 1)) == 0:
        return []
    bins = N // 2 + 1 if onesided else N
    return mm_launches(M, N, bins, F32, F32, native_bf16)


def dft_c2r_launches(M: int, bins: int, N: int, native_bf16: bool) -> List[Launch]:
    """`fft_c2r_wrapper` -> `_dft_c2r`, any length: the complex64 spectrum's real and imaginary
    parts [M, bins] times the fp32 inverse matrices [bins, N] (two `mm`, one key)."""
    return mm_launches(M, bins, N, F32, F32, native_bf16)


# ---------------------------------------------------------------------------------------------
# Attention (scaled_dot_product_attention): the route, then the math route's two bmm launches.
# ---------------------------------------------------------------------------------------------

def sdpa_operand_dtypes(q: NBXDtype, k: NBXDtype, v: NBXDtype,
                        q_round: Optional[NBXDtype] = None):
    """The (q, k, v) dtypes `scaled_dot_product_attention_wrapper` computes attention with: tl.dot
    on V100 needs matching operands, so three that disagree are all cast to fp32 (Q judged at its
    KV-cache rounding `q_round` when the cache asks for one). Returns (q, k, v, q_round)."""
    if not ((k if q_round is not None else q) == k == v):
        return F32, F32, F32, None
    return q, k, v, q_round


def decode_vec_takes(headdim: int, headdim_v: int, mask_numel: Optional[int], seqlen_k: int) -> bool:
    """Whether a single-query attention on the math route runs the vector decode kernel (no
    autotuned key) — `_try_decode_vec`'s contract: equal head dims, and a mask that is one bias
    per key (or none). Outside it, the math route's two bmm launches."""
    if headdim != headdim_v:
        return False
    return mask_numel is None or mask_numel == seqlen_k


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


def flash_headdim_detour(D: int, enabled: bool = True) -> int:
    """The head dim the flash route runs at: a power of two >= 128 is zero-padded by one
    (`scaled_dot_product_attention_wrapper`'s detour around the flash kernel's wrong answers at
    those dims), and the padded call re-enters the wrapper — routed afresh at D + 1 (and V's head
    dim + 1). `enabled` False is the wrapper's diagnostic `NBX_D128_DETOUR=0`."""
    return D + 1 if (enabled and D >= 128 and (D & (D - 1)) == 0) else D


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


class ConvRowOverBand(ValueError):
    """A convolution whose single output row is over the band budget: band streaming cannot split
    it, and the per-band recursion would call itself on the same row without end."""


def conv2d_band_rows(N: int, out_c: int, out_h: int, out_w: int, out_bytes: int, band_bytes: int) -> int:
    """The output rows of one band of `conv2d_wrapper`'s band streaming (`_conv2d_band_streamed`):
    bands of about half the budget, evenly cut. A band that cannot be smaller than the output —
    one row already over the budget — is refused by name: the recursion it would start calls the
    same row forever (a 1-row codec conv, found by the derived census 2026-09-29)."""
    band_target = max(1, band_bytes // 2)
    row_bytes = N * out_c * out_w * out_bytes
    rows_per_band = max(1, band_target // max(1, row_bytes))
    tile_factor = max(1, (out_h + rows_per_band - 1) // rows_per_band)
    band_oh = (out_h + tile_factor - 1) // tile_factor
    if band_oh >= out_h:
        raise ConvRowOverBand(
            f"conv2d output [{N}, {out_c}, {out_h}, {out_w}]: one row is {row_bytes} bytes, over "
            f"the band budget {band_bytes}; band streaming cannot split it")
    return band_oh


def conv2d_launches(N: int, in_c: int, in_h: int, in_w: int, out_c: int, kh: int, kw: int,
                    sh: int, sw: int, ph: int, pw: int, dh: int, dw: int, groups: int,
                    x: NBXDtype, w: NBXDtype, compute: Optional[NBXDtype], band_bytes: int,
                    depthwise_enabled: bool = True) -> List[Launch]:
    """`conv2d_wrapper` on a 4-D weight: narrowing; the depthwise stencil when groups == in_c ==
    out_c at dilation 1; band streaming (per-band recursion at the original padding) when the
    output, sized at the INPUT's dtype, exceeds `band_bytes`; otherwise one `conv2d_forward_kernel`
    launch. The output dtype is the run's compute dtype when set (`_NBX_COMPUTE_DTYPE`), else the
    input's. The `fp16` key flag names an fp16 input (fixed 2026-09-29, aaad1c48; the entries were
    re-keyed in place)."""
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
        band_oh = conv2d_band_rows(N, out_c, out_h, out_w, xb, band_bytes)
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
                      x == F16, tag(x), tag(w), tag(out)))]


def tiled_conv2d_bands(IH: int, out_h: int, kh: int, sh: int, dh: int, pad_h: int, tile_factor: int):
    """The bands of Prism's op-level tiled conv (`_tiled_conv2d_spatial_nbx`, real halo): per band
    (oh_start, oh_end, in_start, in_end, pad_top, pad_bot, skip) — the output rows, the input
    rows read (clamped), the image-edge padding added, the band's leading conv rows to skip.
    The image-edge padding is `max(0, -read_start)` alone: `read_start` already carries -pad_h
    (P-NBX-TILED-CONV2D-SMALL-SCALE 2026-05-14 — adding pad_h again on the edge bands shifted the
    first and last bands by pad_h rows, cos near 0 against F.conv2d at kh >= 3, pad_h >= 1)."""
    tf = max(1, int(tile_factor))
    band_oh = (out_h + tf - 1) // tf
    # The halo in input rows, rounded up to a whole number of strides: the band's first conv row
    # must be an output row, so the rows skipped on the read side are halo // stride. At stride 1
    # this is the halo it always was; at stride 2 a 1-row halo misaligned every internal band by
    # half an output row (found by this function's test, 2026-09-29 — Prism tiles strided convs).
    halo_h = -(-((kh - 1) * dh // 2) // sh) * sh
    bands = []
    for oh_start in range(0, out_h, band_oh):
        oh_end = min(oh_start + band_oh, out_h)
        halo_top = 0 if oh_start == 0 else halo_h
        halo_bot = 0 if oh_end == out_h else halo_h
        read_start = oh_start * sh - pad_h - halo_top
        read_end = (oh_end - 1) * sh + dh * (kh - 1) + 1 - pad_h + halo_bot
        start, end = max(0, read_start), min(IH, read_end)
        if end <= start:
            continue
        bands.append((oh_start, oh_end, start, end, max(0, -read_start), max(0, read_end - IH),
                      halo_top // sh))
    return bands


def tiled_conv2d_launches(N: int, in_c: int, IH: int, IW: int, out_c: int, kh: int, kw: int,
                          sh: int, sw: int, ph: int, pw: int, dh: int, dw: int, groups: int,
                          x: NBXDtype, w: NBXDtype, compute: Optional[NBXDtype], band_bytes: int,
                          tile_factor: int) -> List[Launch]:
    """The launches of an op-level tiled conv: per band of `tiled_conv2d_bands`, the band's rows
    (padded by the image-edge rows, and by pad_w on both sides whenever any padding applies) run
    through `conv2d_wrapper` at padding 0."""
    out_h = (IH + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    out: List[Launch] = []
    for _o0, _o1, start, end, pt, pb, _ht in tiled_conv2d_bands(IH, out_h, kh, sh, dh, ph, tile_factor):
        padded = pt > 0 or pb > 0 or pw > 0
        bh = end - start + pt + pb
        bw = IW + 2 * pw if padded else IW
        for l in conv2d_launches(N, in_c, bh, bw, out_c, kh, kw, sh, sw, 0, 0, dh, dw, groups,
                                 x, w, compute, band_bytes):
            if l not in out:
                out.append(l)
    return out


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
