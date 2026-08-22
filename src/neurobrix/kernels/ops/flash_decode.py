"""Flash-decoding — fused decode attention (T_q == 1), pure @triton.jit.

The decode step's attention on the Triton path was `_math_attention`:
bmm(QK^T) -> softmax -> bmm(PV) with the scores materialised. On the GPU
timeline of the captured step, that chain is the LARGEST single kernel
cost at a working context (baddbmm 9.3 ms/token at 4,164) and it grows
linearly with T_k — it is the long-context term.

This is the flash-decoding shape (Stanford CRFM 2023): split the KV
sequence across programs so the SMs fill even at batch 1, each split
computing a partial output with ONLINE SOFTMAX plus its running max and
sum-of-exponentials, then a SECOND kernel combines the splits.

DETERMINISM IS A DESIGN REQUIREMENT, not an accident. The replay engine
verifies frozen plans byte-equal before adopting them, so a kernel that
varies run-to-run is unusable there. Two rules enforce it:
  - each split writes its own partial slice — no atomics anywhere;
  - the reduction iterates splits in a FIXED ascending order in a single
    program per (batch, kv-head), so the floating-point combination
    order is a constant of the shapes.

GQA is handled the way `_math_attention` now does it: Q's heads are
grouped INSIDE each KV head — `[B, H, 1, D]` viewed as
`[B, H_kv, GROUPS, D]` is a pure view — so K and V are read in their
native cache layout with no broadcast and no materialisation.

HEAD-DIM 128 HAZARD, named at birth: the flash *forward* kernel returns
wrong answers when its head-dim specialisation is exactly 128 or 256 (a
Triton codegen defect, detoured by zero-padding in the wrapper). This
kernel compiles the same class of specialisation, so it is born with a
float64 oracle at D = 127, 128 AND 129 (`test_flash_decode_oracle.py`)
and does not ship a single shape untested against the truth.
"""

import triton
import triton.language as tl


@triton.jit
def flash_decode_split_kernel(
    q_ptr,          # [B*H_kv, GROUPS, D]  (grouped view of q, fp16)
    k_ptr,          # [B*H_kv, T_k, D]     (native cache layout, fp16)
    v_ptr,          # [B*H_kv, T_k, D]
    bias_ptr,       # [T_k] additive fp32 (or unused when HAS_BIAS=0)
    opart_ptr,      # [B*H_kv, SPLIT, GROUPS, D] fp32
    mpart_ptr,      # [B*H_kv, SPLIT, GROUPS] fp32
    lpart_ptr,      # [B*H_kv, SPLIT, GROUPS] fp32
    T_k, seg_len,
    sm_scale,
    stride_qh, stride_qg,
    stride_kh, stride_kn,
    stride_oh, stride_os, stride_og,
    stride_mh, stride_ms,
    GROUPS: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_h = tl.program_id(0)        # b * H_kv + h_kv
    pid_s = tl.program_id(1)        # split index

    offs_g = tl.arange(0, BLOCK_G)
    offs_d = tl.arange(0, BLOCK_D)
    mask_g = offs_g < GROUPS
    mask_d = offs_d < D

    # Q tile for this kv-head: [BLOCK_G, BLOCK_D]
    q = tl.load(q_ptr + pid_h * stride_qh + offs_g[:, None] * stride_qg
                + offs_d[None, :],
                mask=mask_g[:, None] & mask_d[None, :], other=0.0)

    seg_start = pid_s * seg_len
    seg_end = tl.minimum(seg_start + seg_len, T_k)

    m_i = tl.full([BLOCK_G], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_G], dtype=tl.float32)
    acc = tl.zeros([BLOCK_G, BLOCK_D], dtype=tl.float32)

    for n0 in range(seg_start, seg_end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seg_end

        k_tile = tl.load(k_ptr + pid_h * stride_kh
                         + offs_n[:, None] * stride_kn + offs_d[None, :],
                         mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        # scores [BLOCK_G, BLOCK_N] in fp32
        s = tl.dot(q, tl.trans(k_tile))
        s = s.to(tl.float32) * sm_scale
        if HAS_BIAS:
            b = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
            s = s + b[None, :].to(tl.float32)
        # out-of-segment lanes must never win the max nor add to the sum
        s = tl.where(mask_n[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        # a fully -inf row (empty segment tail) keeps m_new=-inf; the
        # exp arguments below become NaN through inf-inf, so pin them.
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        corr = tl.exp(tl.where(m_i == float("-inf"), float("-inf"),
                               m_i - m_safe))
        p = tl.exp(s - m_safe[:, None])
        p = tl.where(mask_n[None, :], p, 0.0)

        v_tile = tl.load(v_ptr + pid_h * stride_kh
                         + offs_n[:, None] * stride_kn + offs_d[None, :],
                         mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        acc = acc * corr[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
        l_i = l_i * corr + tl.sum(p, axis=1)
        m_i = m_new

    base_o = (opart_ptr + pid_h * stride_oh + pid_s * stride_os
              + offs_g[:, None] * stride_og + offs_d[None, :])
    tl.store(base_o, acc, mask=mask_g[:, None] & mask_d[None, :])
    base_m = mpart_ptr + pid_h * stride_mh + pid_s * stride_ms + offs_g
    tl.store(base_m, m_i, mask=mask_g)
    base_l = lpart_ptr + pid_h * stride_mh + pid_s * stride_ms + offs_g
    tl.store(base_l, l_i, mask=mask_g)


@triton.jit
def flash_decode_reduce_kernel(
    opart_ptr, mpart_ptr, lpart_ptr,
    out_ptr,        # [B*H_kv, GROUPS, D] output dtype
    stride_oh, stride_os, stride_og,
    stride_mh, stride_ms,
    stride_outh, stride_outg,
    SPLIT: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D: tl.constexpr,
):
    """Combine the split partials via log-sum-exp correction.

    One program per (batch, kv-head); splits iterated in FIXED ascending
    order — the combination order is a constant, hence deterministic.
    """
    pid_h = tl.program_id(0)
    offs_g = tl.arange(0, BLOCK_G)
    offs_d = tl.arange(0, BLOCK_D)
    mask_g = offs_g < GROUPS
    mask = mask_g[:, None] & (offs_d[None, :] < D)

    m_i = tl.full([BLOCK_G], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_G], dtype=tl.float32)
    acc = tl.zeros([BLOCK_G, BLOCK_D], dtype=tl.float32)

    for s in range(SPLIT):
        m_s = tl.load(mpart_ptr + pid_h * stride_mh + s * stride_ms + offs_g,
                      mask=mask_g, other=float("-inf"))
        l_s = tl.load(lpart_ptr + pid_h * stride_mh + s * stride_ms + offs_g,
                      mask=mask_g, other=0.0)
        o_s = tl.load(opart_ptr + pid_h * stride_oh + s * stride_os
                      + offs_g[:, None] * stride_og + offs_d[None, :],
                      mask=mask, other=0.0)
        m_new = tl.maximum(m_i, m_s)
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        c_old = tl.exp(tl.where(m_i == float("-inf"), float("-inf"),
                                m_i - m_safe))
        c_new = tl.exp(tl.where(m_s == float("-inf"), float("-inf"),
                                m_s - m_safe))
        acc = acc * c_old[:, None] + o_s * c_new[:, None]
        l_i = l_i * c_old + l_s * c_new
        m_i = m_new

    # A row whose every key was masked has l == 0: emit 0, matching the
    # fully-masked-row guard of both existing SDPA paths (nan_to_num).
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / l_safe[:, None]
    tl.store(out_ptr + pid_h * stride_outh + offs_g[:, None] * stride_outg
             + offs_d[None, :], out, mask=mask)
