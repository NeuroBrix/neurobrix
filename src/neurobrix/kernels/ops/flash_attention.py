"""Flash Attention forward — pure @triton.jit kernel.

Extracted from Dao-AILab flash-attention (flash_attn_triton.py).
The gold standard Triton Flash Attention implementation.

Features:
- Causal and non-causal attention
- Self-attention and cross-attention
- Arbitrary seqlens (not just multiples of 128)
- Head dimensions up to 128
- Attention bias (vector or matrix) — REQUIRED, not optional

Tested on A100. See original source for caveats about race conditions
on non-64/128 head dimensions.

BIAS_TYPE accepts only "vector" or "matrix" — the original "none"
path (zero bias materialized in registers via tl.zeros) was removed
because Triton's IR-level optimization passes propagate the bias
*origin* (constant-fold candidate vs memory-to-register load) through
to the MMA selection of the downstream tl.dot, producing
non-bit-equivalent results vs the matrix path on identical Q/K/V in
fp32 inputs. Callers that previously passed BIAS_TYPE="none" must now
provide a memory-resident zero bias buffer (the Python wrapper handles
this transparently via a cached zero buffer).
"""

import triton
import triton.language as tl


@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def flash_attention_forward_kernel(
    Q, K, V, Bias,
    Out, Lse, TMP,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_bb, stride_bh, stride_bm,
    stride_ob, stride_oh, stride_om,
    nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim,
    CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GQA_GROUPS: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # GQA: each Q head maps to K/V head (off_h // GQA_GROUPS). For plain MHA
    # pass GQA_GROUPS=1 → off_h_kv == off_h, zero overhead.
    off_h_kv = off_h // GQA_GROUPS

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h_kv * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h_kv * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    else:  # BIAS_TYPE == "matrix"
        b_ptrs = (
            Bias + off_b * stride_bb + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )

    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    # Load Q — stays in SRAM throughout
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)

    # Loop over K, V blocks. Causal masking is applied via the bias
    # tensor (memory-loaded), not via an internal IS_CAUSAL constexpr —
    # the wrapper materializes a causal additive mask {-inf above diag,
    # 0 elsewhere} when needed and passes it as bias_matrix. This keeps
    # every tensor reaching tl.dot on a single tl.load path.
    end_n = seqlen_k
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Load K
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn,
                            mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)

        # QK^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        if not EVEN_N:
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))

        # Bias load — always memory-resident (BIAS_TYPE is "vector" or
        # "matrix", never "none"). Triton's IR-level optimizer propagates
        # the bias *origin* (constant-fold candidate vs memory load) all
        # the way to the MMA selection of the downstream tl.dot, producing
        # non-bit-equivalent results when the bias originates from
        # tl.zeros vs tl.load. The Python wrapper guarantees a memory-
        # resident bias by routing no-mask calls through a cached zero
        # buffer (see scaled_dot_product_attention_wrapper). Single
        # softmax path → single tl.dot compile → bit-equivalent results
        # across all NVIDIA archs (Volta SIMT, Ampere/Hopper TF32) and
        # AMD ROCm CDNA matrix cores.
        if BIAS_TYPE == "vector":
            if EVEN_N:
                bias = tl.load(b_ptrs + start_n).to(tl.float32)
            else:
                bias = tl.load(b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0).to(tl.float32)
            bias = bias[None, :]
        else:  # BIAS_TYPE == "matrix"
            if EVEN_M & EVEN_N:
                bias = tl.load(b_ptrs + start_n).to(tl.float32)
            else:
                bias = tl.load(b_ptrs + start_n,
                               mask=(offs_m[:, None] < seqlen_q) & ((start_n + offs_n)[None, :] < seqlen_k),
                               other=0.0).to(tl.float32)
        qk = qk * softmax_scale + bias
        m_ij = tl.maximum(tl.max(qk, 1), lse_i)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # Scale accumulator
        acc_o_scale = tl.exp(m_i - m_ij)
        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]

        # Load V and accumulate
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn,
                            mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # Update statistics
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    # Final output scaling
    o_scale = tl.exp(m_i - lse_i)
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]

    # Store LSE and output
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)

    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out + off_b * stride_ob + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim))
