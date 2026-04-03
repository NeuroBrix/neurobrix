"""LayerNorm — pure @triton.jit kernel. Ported from attorch (MIT)."""

import triton
import triton.language as tl
from triton import next_power_of_2

from ._configs import batch_block_heuristic


@triton.heuristics({
    'BLOCK_SIZE_BATCH': batch_block_heuristic,
    'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim']),
})
@triton.jit
def layer_norm_forward_kernel(
    input_ptr, weight_ptr, bias_ptr,
    mean_ptr, inv_std_ptr, output_ptr,
    batch_dim, feat_dim,
    input_batch_stride, input_feat_stride,
    output_batch_stride, output_feat_stride,
    eps,
    scale_by_weight: tl.constexpr,
    add_bias: tl.constexpr,
    save_stats: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_FEAT: tl.constexpr,
):
    """Layer normalization forward.

    input: [batch_dim, feat_dim]
    weight: [feat_dim] (optional)
    bias: [feat_dim] (optional)
    output: [batch_dim, feat_dim]
    mean, inv_std: [batch_dim] (optional, for backward)
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
    mean = tl.sum(inp, axis=1) / feat_dim
    diff = tl.where(feat_mask[None, :], inp - mean[:, None], 0)
    inv_std = tl.rsqrt(tl.sum(diff * diff, axis=1) / feat_dim + eps)

    if save_stats:
        tl.store(mean_ptr + batch_offset, mean, mask=batch_mask)
        tl.store(inv_std_ptr + batch_offset, inv_std, mask=batch_mask)

    output = diff * inv_std[:, None]
    if scale_by_weight:
        weight = tl.load(weight_ptr + feat_offset, mask=feat_mask)
        output *= weight
        if add_bias:
            bias = tl.load(bias_ptr + feat_offset, mask=feat_mask)
            output += bias

    tl.store(output_ptr, output,
             mask=batch_mask[:, None] & feat_mask[None, :])
