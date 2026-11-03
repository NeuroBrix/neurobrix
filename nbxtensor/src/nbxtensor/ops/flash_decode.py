"""The fixed-order reduce of split-KV decode attention (T_q == 1), pure @triton.jit.

Decode attention splits the KV sequence across programs so the SMs fill even at batch 1; each split writes a
partial output with its running max and sum of exponentials (ONLINE SOFTMAX), and this kernel combines the splits.
Its users are the vector decode kernels (`ops/decode_attn_vec.py`, `ops/paged_decode_attn.py`).

DETERMINISM IS A DESIGN REQUIREMENT. The replay engine verifies frozen plans byte-equal before adopting them, so a
kernel that varies run to run is unusable there: each split writes its own partial slice (no atomics) and the
reduction iterates the splits in a FIXED ascending order in one program per (batch, kv-head), so the floating-point
combination order is a constant of the shapes.

The tensor-core split kernel that once shared this file (`flash_decode_split_kernel`) was served only behind an
environment switch and measured slower than the vector kernels; it was removed with its route (2026-11-03).
"""

import triton
import triton.language as tl


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
    pid_h = tl.program_id(0).to(tl.int64)
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
