"""RoPE (Rotary Position Embedding) — pure @triton.jit kernels.

Fused forward kernel: applies rotary embedding in-place to Q and K.
HuggingFace Llama/Mistral style: split-half rotation (not interleaved).
Extracted from Liger-Kernel (Apache 2.0).

Layout:
    q: (bsz, seq_len, n_q_heads, head_dim)  — physical (transposed from [B,H,S,D])
    k: (bsz, seq_len, n_kv_heads, head_dim)
    cos: (1|bsz, seq_len, head_dim)
    sin: (1|bsz, seq_len, head_dim)

Grid: (bsz * seq_len,)
Each program handles one (batch, position) across ALL heads.
"""

import triton
import triton.language as tl


@triton.jit
def rope_forward_kernel(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    cos,
    cos_row_stride,
    sin,
    sin_row_stride,
    sl,
    bs: tl.constexpr,
    cos_bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BACKWARD_PASS: tl.constexpr = False,
):
    """Fused RoPE kernel — applies rotary embedding in-place to Q and K.

    Each program instance processes one (batch, seq_position) pair across
    all Q heads and K heads simultaneously. The rotation formula is:

        y_left  = x_left  * cos - x_right * sin
        y_right = x_right * cos + x_left  * sin

    For backward pass, sin is negated (equivalent to inverse rotation).
    """
    pid = tl.program_id(0).to(tl.int64)

    # Advance pointers to current (batch, position) row
    q_ptr = q_ptr + pid * q_row_stride
    k_ptr = k_ptr + pid * k_row_stride

    # Locate cos/sin row for this token position
    # pid indexes [bsz, seq_len] flattened; seq_len is fastest-changing
    batch_idx = pid // sl
    cos_row_idx = pid % sl
    cos = cos + tl.where(
        cos_bs == 1,
        cos_row_idx * cos_row_stride,
        batch_idx * (sl * cos_row_stride) + cos_row_idx * cos_row_stride,
    )
    sin = sin + tl.where(
        cos_bs == 1,
        cos_row_idx * sin_row_stride,
        batch_idx * (sl * sin_row_stride) + cos_row_idx * sin_row_stride,
    )

    # Load cos/sin (only need left half — right half is identical)
    cos_offsets = tl.arange(0, pad_hd // 2)
    cos_mask = cos_offsets < hd // 2
    # A table WIDER than Q is rounded to Q's dtype on load (round-to-nearest,
    # the conversion a stored cast makes): the wrapper no longer materialises
    # a cast copy of the step's fp32 cos/sin per layer for an fp16 Q. A table
    # narrower than Q is widened by the wrapper beforehand (a copy, said in
    # its line): widened on load, the fp32 rotation's fused multiply-add
    # moved with the convert before it (Sana's Gemma-2 encoder, 2026-09-07).
    cos_row = tl.load(cos + cos_offsets, mask=cos_mask, other=0).to(q_ptr.dtype.element_ty)
    sin_row = tl.load(sin + cos_offsets, mask=cos_mask, other=0).to(q_ptr.dtype.element_ty)

    # --- Q heads: load left half and right half ---
    first_half_q_offsets = (
        tl.arange(0, pad_n_qh)[:, None] * hd
        + tl.arange(0, pad_hd // 2)[None, :]
    )
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (
        tl.arange(0, pad_hd // 2)[None, :] < hd // 2
    )
    q_tile_1 = tl.load(
        q_ptr + first_half_q_offsets, mask=first_q_mask, other=0
    ).to(sin_row.dtype)

    second_half_q_offsets = first_half_q_offsets + (hd // 2)
    q_tile_2 = tl.load(
        q_ptr + second_half_q_offsets, mask=first_q_mask, other=0
    ).to(sin_row.dtype)

    # --- K heads: load left half and right half ---
    first_half_k_offsets = (
        tl.arange(0, pad_n_kh)[:, None] * hd
        + tl.arange(0, pad_hd // 2)[None, :]
    )
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (
        tl.arange(0, pad_hd // 2)[None, :] < hd // 2
    )
    k_tile_1 = tl.load(
        k_ptr + first_half_k_offsets, mask=first_k_mask, other=0
    ).to(sin_row.dtype)

    second_half_k_offsets = first_half_k_offsets + (hd // 2)
    k_tile_2 = tl.load(
        k_ptr + second_half_k_offsets, mask=first_k_mask, other=0
    ).to(sin_row.dtype)

    # --- Apply rotation ---
    if not BACKWARD_PASS:
        # Forward: y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
        # The plain expression, as the backend contracts it per dtype: written
        # with an explicit fma it matched the fp32 path but not the fp16 one
        # (VibeVoice, 2026-09-07) — the contraction is not the same in both,
        # and the bytes of every certified row depend on it staying as it is.
        new_q_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
        new_q_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
        new_k_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
        new_k_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    else:
        # Backward (inverse rotation): negate sin
        new_q_1 = q_tile_1 * cos_row + q_tile_2 * sin_row
        new_q_2 = q_tile_2 * cos_row - q_tile_1 * sin_row
        new_k_1 = k_tile_1 * cos_row + k_tile_2 * sin_row
        new_k_2 = k_tile_2 * cos_row - k_tile_1 * sin_row

    # Store results in-place
    tl.store(q_ptr + first_half_q_offsets, new_q_1, mask=first_q_mask)
    tl.store(q_ptr + second_half_q_offsets, new_q_2, mask=first_q_mask)
    tl.store(k_ptr + first_half_k_offsets, new_k_1, mask=first_k_mask)
    tl.store(k_ptr + second_half_k_offsets, new_k_2, mask=first_k_mask)
