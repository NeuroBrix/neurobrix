"""Max reduction — pure @triton.jit kernel, the row walked in tiles of at most REDUCTION_FEAT_TILE."""

import triton
import triton.language as tl

from ._configs import batch_block_heuristic, reduction_feat_tile


@triton.heuristics({
    'BLOCK_SIZE_BATCH': batch_block_heuristic,
    'BLOCK_SIZE_FEAT': reduction_feat_tile,
})
@triton.jit
def amax_forward_kernel(
    input_ptr, output_ptr,
    batch_dim, feat_dim,
    input_batch_stride, input_feat_stride,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_FEAT: tl.constexpr,
):
    """Max reduction over the last dimension: input [batch_dim, feat_dim] -> output [batch_dim].

    The first tile is reduced alone and every further tile combined into it, so a row that fits one
    tile is reduced by exactly the instructions it was before the loop existed.
    """
    batch_pid = tl.program_id(0).to(tl.int64)
    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    feat_offset = tl.arange(0, BLOCK_SIZE_FEAT)
    batch_mask = batch_offset < batch_dim
    row_ptr = input_ptr + input_batch_stride * batch_offset[:, None]

    feat_mask = feat_offset < feat_dim
    inp = tl.load(row_ptr + input_feat_stride * feat_offset[None, :],
                  mask=batch_mask[:, None] & feat_mask[None, :], other=-float('inf'))
    result = tl.max(inp, axis=1)
    for start in range(BLOCK_SIZE_FEAT, feat_dim, BLOCK_SIZE_FEAT):
        offs = start + feat_offset
        feat_mask = offs < feat_dim
        inp = tl.load(row_ptr + input_feat_stride * offs[None, :].to(tl.int64),
                      mask=batch_mask[:, None] & feat_mask[None, :], other=-float('inf'))
        result = tl.maximum(result, tl.max(inp, axis=1))
    tl.store(output_ptr + batch_offset, result, mask=batch_mask)
