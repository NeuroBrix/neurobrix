"""Vector (SIMT) decode attention — Tq == 1, pure @triton.jit, no tl.dot.

Second attempt at the decode-attention bar (~0.5 ms/token at 4k = the
KV bytes at HBM speed). The first attempt (ops/flash_decode.py) kept
the flash-decoding two-pass SPLIT structure but computed QK/PV with
`tl.dot` on a 16-row Q tile carrying 4 real GQA rows — and lost on the
row three times. The 2026-08-23 web research (R16) showed the split
was never the problem and the dot was: every production Tq=1 kernel on
pre-Ampere silicon is a SIMT kernel — FasterTransformer's
masked_multihead_attention (per-thread fp16 FMA + shuffle reduce, no
MMA anywhere), vLLM's paged_attention v1/v2 (16-byte vectorized K
loads, fixed-order v2 partition reduce), llama.cpp's fattn-vec/tile
(picked over MMA on Volta at exactly our effective batch), lightllm's
token_attention (`tl.sum(q[None, :] * k, 1)`, BLOCK_N=32, no dot).
Triton lowers fp16 `tl.dot` to FFMA on sm_70 (no HMMA lowering — the
Phase 1.5 measurement), so the MMA tile staging is pure cost there.

Structure here (the lightllm/vLLM shape):
  - grid = (B * H_q, split): ONE QUERY ROW per program — no G tile at
    all. The GQA K re-read (4 Q heads share a KV head) is absorbed by
    L2 temporal locality, exactly as in vLLM/FT which organise the
    same way.
  - QK: `tl.sum(q[None, :] * k_tile, 1)` — an FMA reduction over D,
    product tile (BLOCK_N, BLOCK_D) only.
  - online softmax with SCALAR m/l state, fp32 accumulator [BLOCK_D].
  - split-KV partials (out, m, l) merged by the EXISTING fixed-order
    `flash_decode_reduce_kernel` (ops/flash_decode.py) viewed at
    GROUPS=1 — deterministic by construction, no atomics, proven.
  - K/V are read by their OWN strides: the bucketed KV cache's
    prefix-slice views flow in directly (the strided-KV lot's
    contract), no materialisation.
  - num_stages=1 (sm_70 has no async-copy engine to pipeline with).

Born with its float64 oracle at D = 127, 128 AND 129
(test_decode_vec_oracle.py) and a route ACTIVATION proof — the
2026-08-23 lesson: a route-dependent claim needs the route pinned.
"""

import triton
import triton.language as tl

from nbxtensor.ops._configs import nbx_autotune


@triton.jit
def _round_q(q, Q_TO: tl.constexpr, Q_SATURATE: tl.constexpr):
    """Q as the caller asked it read: rounded to Q_TO (the KV cache's
    dtype) with the stored cast's protected fp16 conversion when
    Q_SATURATE, then widened to the fp32 the dot runs in — the bytes of a
    cast copy of Q, without the copy."""
    if Q_TO is not None:
        f = q.to(tl.float32)
        if Q_SATURATE:
            is_finite = (f == f) & (f != float("inf")) & (f != float("-inf"))
            clamped = tl.minimum(tl.maximum(f, -65504.0), 65504.0)
            f = tl.where(is_finite, clamped, f)
        q = f.to(Q_TO)
    return q.to(tl.float32)


# The key block and warps of the vector decode, candidates the certification measures per key (the judged row,
# 2026-08-23: 32 keys a step, 4 warps, among them); one stage — the loop streams K and V once.
_DECODE_VEC_CONFIGS = [triton.Config({'BLOCK_N': bn}, num_warps=w, num_stages=1)
                       for bn in (16, 32, 64, 128) for w in (1, 2, 4, 8)]


@nbx_autotune(configs=_DECODE_VEC_CONFIGS, key=['T_BUCKET', 'D', 'D_V', 'GQA_GROUPS', 'HAS_BIAS'], cache_results=True)
@triton.jit
def decode_attn_vec_split_kernel(
    q_ptr,          # [B*H_q, D]        one query row per program (fp16)
    k_ptr,          # [B*H_kv, T_k, D]  native cache layout, own strides
    v_ptr,          # [B*H_kv, T_k, D_v]
    bias_ptr,       # [T_k] additive fp32/fp16 (unused when HAS_BIAS=0)
    opart_ptr,      # [B*H_q, SPLIT, D_v] fp32 partial outputs
    mpart_ptr,      # [B*H_q, SPLIT] fp32 running max per split
    lpart_ptr,      # [B*H_q, SPLIT] fp32 sum-of-exp per split
    T_k, seg_len,
    T_BUCKET,       # T_k's bucket: the autotune key's length class (unused in the body)
    sm_scale,
    stride_qh,
    stride_kh, stride_kn,
    stride_vh, stride_vn,
    stride_oh, stride_os,
    stride_mh,
    GQA_GROUPS: tl.constexpr,   # H_q // H_kv — maps Q head -> KV head
    BLOCK_D: tl.constexpr,
    D: tl.constexpr,
    D_V: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    Q_TO: tl.constexpr = None,        # a dtype: Q is rounded to it on load (the cache's)
    Q_SATURATE: tl.constexpr = False, # the protected fp16 rounding of that cast
    BLOCK_N: tl.constexpr = 32,
):
    pid_q = tl.program_id(0).to(tl.int64)        # b * H_q + h_q
    pid_s = tl.program_id(1).to(tl.int64)        # split index
    pid_kv = pid_q // GQA_GROUPS    # owning KV head row

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    mask_dv = offs_d < D_V

    q = tl.load(q_ptr + pid_q * stride_qh + offs_d,
                mask=mask_d, other=0.0)
    q = _round_q(q, Q_TO, Q_SATURATE)

    seg_start = pid_s * seg_len
    seg_start = seg_start.to(tl.int32)   # loop bounds are 32-bit counts: the Metal lowering refuses a 64-bit scf.for bound (addressing stays 64-bit through the program ids)
    seg_end = tl.minimum(seg_start + seg_len, T_k).to(tl.int32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for n0 in range(seg_start, seg_end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seg_end

        k_tile = tl.load(k_ptr + pid_kv * stride_kh
                         + offs_n[:, None] * stride_kn + offs_d[None, :],
                         mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        # scores [BLOCK_N] — FMA reduction over D, no tl.dot
        s = tl.sum(q[None, :] * k_tile.to(tl.float32), 1) * sm_scale
        if HAS_BIAS:
            b = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
            s = s + b.to(tl.float32)
        s = tl.where(mask_n, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 0))
        # empty tail keeps m_new=-inf; pin the exp arguments (the
        # flash_decode kernel's own guard, scalar form)
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        corr = tl.exp(tl.where(m_i == float("-inf"), float("-inf"),
                               m_i - m_safe))
        p = tl.exp(s - m_safe)
        p = tl.where(mask_n, p, 0.0)

        v_tile = tl.load(v_ptr + pid_kv * stride_vh
                         + offs_n[:, None] * stride_vn + offs_d[None, :],
                         mask=mask_n[:, None] & mask_dv[None, :], other=0.0)
        acc = acc * corr + tl.sum(p[:, None] * v_tile.to(tl.float32), 0)
        l_i = l_i * corr + tl.sum(p, 0)
        m_i = m_new

    tl.store(opart_ptr + pid_q * stride_oh + pid_s * stride_os + offs_d,
             acc, mask=mask_dv)
    tl.store(mpart_ptr + pid_q * stride_mh + pid_s, m_i)
    tl.store(lpart_ptr + pid_q * stride_mh + pid_s, l_i)


