"""RMSNorm — pure @triton.jit kernel. Ported from attorch (MIT)."""

import triton
import triton.language as tl
from triton import next_power_of_2

from ._configs import batch_block_heuristic


@triton.heuristics({
    'BLOCK_SIZE_BATCH': batch_block_heuristic,
    'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim']),
})
@triton.jit
def rms_norm_forward_kernel(
    input_ptr, weight_ptr,
    output_ptr,
    batch_dim, feat_dim,
    input_batch_stride, input_feat_stride,
    output_batch_stride, output_feat_stride,
    eps,
    scale_by_weight: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_FEAT: tl.constexpr,
):
    """RMS normalization forward.

    input: [batch_dim, feat_dim]
    weight: [feat_dim] (optional)
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
                  mask=batch_mask[:, None] & feat_mask[None, :]).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(inp * inp, axis=1) / feat_dim + eps)
    output = inp * inv_rms[:, None]

    if scale_by_weight:
        weight = tl.load(weight_ptr + feat_offset, mask=feat_mask)
        output *= weight

    tl.store(output_ptr, output,
             mask=batch_mask[:, None] & feat_mask[None, :])
