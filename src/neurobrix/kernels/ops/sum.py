"""Sum reduction — pure @triton.jit kernel."""

import triton
import triton.language as tl
from triton import next_power_of_2

from ._configs import batch_block_heuristic


@triton.heuristics({
    'BLOCK_SIZE_BATCH': batch_block_heuristic,
    'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim']),
})
@triton.jit
def sum_forward_kernel(
    input_ptr, output_ptr,
    batch_dim, feat_dim,
    input_batch_stride, input_feat_stride,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_FEAT: tl.constexpr,
):
    """Sum reduction over last dimension.

    input: [batch_dim, feat_dim] → output: [batch_dim]
    """
    batch_pid = tl.program_id(0)
    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    feat_offset = tl.arange(0, BLOCK_SIZE_FEAT)

    batch_mask = batch_offset < batch_dim
    feat_mask = feat_offset < feat_dim

    input_ptr += (input_batch_stride * batch_offset[:, None] +
                  input_feat_stride * feat_offset[None, :])

    inp = tl.load(input_ptr,
                  mask=batch_mask[:, None] & feat_mask[None, :],
                  other=0.0).to(tl.float32)
    result = tl.sum(inp, axis=1)
    tl.store(output_ptr + batch_offset, result, mask=batch_mask)
