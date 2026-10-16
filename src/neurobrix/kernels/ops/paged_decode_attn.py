"""Paged KV: the append and the decode attention read through a page table (state engine step 2,
docs/design/state-engine.md §4.3; the shared bookkeeping is core/state/pages.py).

The state of a decoding sequence lives in fixed-size pages of a per-node pool, granted as the sequence grows and never
at max_len. One pool per layer and per K / V, laid out [num_pages, kv_heads, page_tokens, D]: the positions of one KV
head inside one page are contiguous rows, so a tile of BLOCK_N positions of one head is one strided block read.

- `paged_kv_append_kernel` writes new positions through the slot mapping (vLLM's `reshape_and_cache` shape): slot =
  page * page_tokens + position-in-page, one int32 per new token, computed by the PageTable.
- `paged_decode_attn_split_kernel` is the vector (SIMT) split-KV decode read of ops/decode_attn_vec.py — one query row
  per program, FMA reduction over D, scalar online softmax, fp32 accumulator, partials merged by the fixed-order
  `flash_decode_reduce_kernel` — with K / V addressed through the block table (one page id per page of the sequence,
  int32, padded with -1). A tile never straddles two pages: every candidate tile divides the page size
  (`PAGED_DECODE_TILE_CANDIDATES`, read by core/state/price.page_tokens) and each split starts on a tile boundary.
  Each sequence of a batch reads its own length (`seq_lens`), so one launch serves a batch of different lengths.

Pure @triton.jit, no tl.dot (sm_70: the vector kernel's measured structure), one source for every backend.
"""

import triton
import triton.language as tl


# The position tiles the paged read is compiled for: powers of two, each dividing the largest, so a page sized from the
# largest holds a whole number of tiles of every candidate. 32 is the vector decode kernel's judged tile (its locked
# protocol of 2026-08-23: seg 256 / BLOCK_N 32 / 4 warps won at all three contexts, ops/decode_attn_vec.py,
# kernels/wrappers.py `_decode_attn_vec`); 16 and 64 are its neighbours, the range the certified directory chooses in.
PAGED_DECODE_TILE_CANDIDATES = (16, 32, 64)

# The split shape: positions per split and the most splits one launch makes — the vector decode kernel's judged config
# (its locked protocol of 2026-08-23, the defaults of `_decode_attn_vec`'s NBX_DV_SEG and its 64-split cap), so the
# paged read partitions a sequence exactly as the contiguous read it replaces.
PAGED_DECODE_SPLIT_POSITIONS = 256
PAGED_DECODE_MAX_SPLITS = 64


@triton.jit
def paged_kv_append_kernel(
    new_ptr,         # [N_tok, H_kv, D] the new positions' K (or V), any float dtype
    pool_ptr,        # [num_pages, H_kv, page_tokens, D] the layer's pool
    slot_ptr,        # [N_tok] int32: page * page_tokens + position in page
    page_tokens,
    stride_nt, stride_nh,
    stride_pp, stride_ph, stride_ps,
    BLOCK_D: tl.constexpr,
    D: tl.constexpr,
):
    """One program per (new token, KV head): its D values to the page slot the slot mapping names, cast to the pool's
    dtype on store."""
    pid_t = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    slot = tl.load(slot_ptr + pid_t).to(tl.int64)
    page = slot // page_tokens
    pos = slot % page_tokens
    x = tl.load(new_ptr + pid_t * stride_nt + pid_h * stride_nh + offs_d, mask=mask_d, other=0.0)
    tl.store(pool_ptr + page * stride_pp + pid_h * stride_ph + pos * stride_ps + offs_d,
             x.to(pool_ptr.dtype.element_ty), mask=mask_d)


@triton.jit
def paged_decode_attn_split_kernel(
    q_ptr,           # [B*H_q, D] one query row per program
    k_pool_ptr,      # [num_pages, H_kv, page_tokens, D]
    v_pool_ptr,      # [num_pages, H_kv, page_tokens, D_v]
    block_table_ptr, # [B, max_pages] int32 page ids, -1 past the sequence
    seq_lens_ptr,    # [B] int32 positions each sequence holds
    opart_ptr,       # [B*H_q, SPLIT, D_v] fp32 partial outputs
    mpart_ptr,       # [B*H_q, SPLIT] fp32 running max per split
    lpart_ptr,       # [B*H_q, SPLIT] fp32 sum-of-exp per split
    seg_len,         # positions per split, a multiple of BLOCK_N
    page_tokens,     # positions per page, a multiple of BLOCK_N
    sm_scale,
    H_Q,
    stride_qh,
    stride_kp, stride_kh, stride_ks,
    stride_vp, stride_vh, stride_vs,
    stride_bt,
    stride_oh, stride_os,
    stride_mh,
    GQA_GROUPS: tl.constexpr,   # H_q // H_kv
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D: tl.constexpr,
    D_V: tl.constexpr,
):
    pid_q = tl.program_id(0).to(tl.int64)        # b * H_q + h_q
    pid_s = tl.program_id(1).to(tl.int64)        # split index
    b = pid_q // H_Q
    h_kv = (pid_q % H_Q) // GQA_GROUPS

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    mask_dv = offs_d < D_V
    q = tl.load(q_ptr + pid_q * stride_qh + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    T_b = tl.load(seq_lens_ptr + b).to(tl.int32)
    seg_start = (pid_s * seg_len).to(tl.int32)   # 32-bit loop bounds (the Metal lowering refuses a 64-bit scf.for bound)
    seg_end = tl.minimum(seg_start + seg_len, T_b).to(tl.int32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for n0 in range(seg_start, seg_end, BLOCK_N):
        # the tile's page: n0 is a multiple of BLOCK_N and BLOCK_N divides page_tokens, so the tile is in one page
        page = tl.load(block_table_ptr + b * stride_bt + n0 // page_tokens).to(tl.int64)
        in_page = (n0 % page_tokens) + tl.arange(0, BLOCK_N)
        mask_n = (n0 + tl.arange(0, BLOCK_N)) < seg_end

        k_tile = tl.load(k_pool_ptr + page * stride_kp + h_kv * stride_kh
                         + in_page[:, None] * stride_ks + offs_d[None, :],
                         mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        s = tl.sum(q[None, :] * k_tile.to(tl.float32), 1) * sm_scale
        s = tl.where(mask_n, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 0))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        corr = tl.exp(tl.where(m_i == float("-inf"), float("-inf"), m_i - m_safe))
        p = tl.exp(s - m_safe)
        p = tl.where(mask_n, p, 0.0)

        v_tile = tl.load(v_pool_ptr + page * stride_vp + h_kv * stride_vh
                         + in_page[:, None] * stride_vs + offs_d[None, :],
                         mask=mask_n[:, None] & mask_dv[None, :], other=0.0)
        acc = acc * corr + tl.sum(p[:, None] * v_tile.to(tl.float32), 0)
        l_i = l_i * corr + tl.sum(p, 0)
        m_i = m_new

    tl.store(opart_ptr + pid_q * stride_oh + pid_s * stride_os + offs_d, acc, mask=mask_dv)
    tl.store(mpart_ptr + pid_q * stride_mh + pid_s, m_i)
    tl.store(lpart_ptr + pid_q * stride_mh + pid_s, l_i)
