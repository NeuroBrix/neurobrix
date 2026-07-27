"""Softmax — pure @triton.jit kernel. Ported from attorch (MIT)."""

import triton
import triton.language as tl
from triton import next_power_of_2

from ._configs import batch_block_heuristic


@triton.heuristics({
    'BLOCK_SIZE_BATCH': batch_block_heuristic,
    'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim']),
})
@triton.jit
def softmax_forward_kernel(
    input_ptr, output_ptr,
    batch_dim, feat_dim,
    input_batch_stride, input_feat_stride,
    output_batch_stride, output_feat_stride,
    log: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_FEAT: tl.constexpr,
):
    """Softmax forward (or log_softmax if log=True).

    input: [batch_dim, feat_dim]
    output: [batch_dim, feat_dim]
    """
    batch_pid = tl.program_id(0)

    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    feat_offset = tl.arange(0, BLOCK_SIZE_FEAT)

    batch_mask = batch_offset < batch_dim
    feat_mask = feat_offset < feat_dim

    input_ptr += (input_batch_stride * batch_offset[:, None] +
                  input_feat_stride * feat_offset[None, :])
    output_ptr += (output_batch_stride * batch_offset[:, None] +
                   output_feat_stride * feat_offset[None, :])

    inp = tl.load(input_ptr,
                  mask=batch_mask[:, None] & feat_mask[None, :],
                  other=-float('inf')).to(tl.float32)

    # Numerical stability: subtract max
    inp -= tl.max(inp, axis=1)[:, None]
    numerator = tl.exp(inp)
    denominator = tl.sum(numerator, axis=1)[:, None]

    if log:
        output = inp - tl.log(denominator)
    else:
        output = numerator / denominator

    tl.store(output_ptr, output,
             mask=batch_mask[:, None] & feat_mask[None, :])


@triton.jit
def softmax_forward_looped_kernel(
    input_ptr, output_ptr,
    feat_dim,
    input_batch_stride, input_feat_stride,
    output_batch_stride, output_feat_stride,
    log: tl.constexpr,
    TILE_FEAT: tl.constexpr,
):
    """Online two-pass softmax for rows too large for one register tile.

    One program per row; the row is scanned in TILE_FEAT chunks with a
    running max / rescaled running sum (pass 1), then normalized and
    stored (pass 2). Register footprint is bounded by TILE_FEAT
    regardless of feat_dim.

    Why this exists: the single-tile kernel above sizes its tile as
    next_power_of_2(feat_dim). At vocab-scale rows (e.g. 151936 →
    262144-element tile) it spills to CUDA local memory, and the driver
    allocates the local pool DEVICE-WIDE at launch — per-thread frame ×
    max resident threads (163,840 on V100) ≈ 2 GiB, measured. The launch
    itself then fails with CUDA_ERROR_OUT_OF_MEMORY whenever the device
    cannot serve that hidden allocation (D9 warm-daemon incident,
    2026-07-27). Bounded tiles keep the frame register-resident, so the
    launch has no local-memory cost at any feat_dim.
    """
    row = tl.program_id(0).to(tl.int64)
    in_row = input_ptr + row * input_batch_stride
    out_row = output_ptr + row * output_batch_stride

    running_max = tl.full((1,), -float('inf'), tl.float32)
    running_sum = tl.zeros((1,), tl.float32)
    for start in range(0, feat_dim, TILE_FEAT):
        offs = start + tl.arange(0, TILE_FEAT)
        mask = offs < feat_dim
        x = tl.load(in_row + offs * input_feat_stride, mask=mask,
                    other=-float('inf')).to(tl.float32)
        new_max = tl.maximum(running_max, tl.max(x, axis=0))
        # -inf guard (online-softmax idiom): while every element seen so
        # far is -inf (top-k/top-p masking routinely -inf-fills whole
        # leading tiles of an unsorted vocab row), new_max stays -inf and
        # both exp arguments would be -inf - (-inf) = NaN, poisoning the
        # running sum for the entire row. Pin alpha to 1 and the shifted
        # tile to -inf (exp → 0) until the first finite value arrives.
        # Fully -inf rows still yield NaN in pass 2 — parity with the
        # single-tile kernel, guarded downstream by the samplers.
        seen_only_neginf = new_max == -float('inf')
        alpha = tl.where(seen_only_neginf, 1.0,
                         tl.exp(running_max - new_max))
        shifted = tl.where(seen_only_neginf, -float('inf'), x - new_max)
        running_sum = running_sum * alpha + tl.sum(tl.exp(shifted), axis=0)
        running_max = new_max

    for start in range(0, feat_dim, TILE_FEAT):
        offs = start + tl.arange(0, TILE_FEAT)
        mask = offs < feat_dim
        x = tl.load(in_row + offs * input_feat_stride, mask=mask,
                    other=-float('inf')).to(tl.float32)
        if log:
            y = x - running_max - tl.log(running_sum)
        else:
            y = tl.exp(x - running_max) / running_sum
        tl.store(out_row + offs * output_feat_stride, y, mask=mask)
