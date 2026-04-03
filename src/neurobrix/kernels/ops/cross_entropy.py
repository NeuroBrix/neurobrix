"""Cross entropy loss — pure @triton.jit kernel.

Extracted from attorch (BobMcDear/attorch) cross_entropy_loss_kernels.py.
Chosen over Liger (523 lines) for inference: simpler, pure, no training extras.
Forward only. Numerically stable log-sum-exp implementation.
"""

import triton
import triton.language as tl
from triton import next_power_of_2

@triton.heuristics({
    'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim']),
})
@triton.jit
def cross_entropy_loss_forward_kernel(
    input_pointer, target_pointer, weight_pointer,
    sum_weights_pointer, output_pointer,
    batch_dim, feat_dim,
    input_batch_stride, input_feat_stride,
    weighted: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_FEAT: tl.constexpr,
):
    """Cross entropy loss: -log(softmax(input)[target]).

    Numerically stable via log-sum-exp trick.

    Args:
        input_pointer: [batch_dim, feat_dim] logits
        target_pointer: [batch_dim] integer class indices
        weight_pointer: optional [feat_dim] class weights
        output_pointer: [batch_dim/BLOCK_SIZE_BATCH] partial losses
    """
    batch_pid = tl.program_id(axis=0)

    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    feat_offset = tl.arange(0, BLOCK_SIZE_FEAT)

    batch_mask = batch_offset < batch_dim
    feat_mask = feat_offset < feat_dim

    target = tl.load(target_pointer + batch_offset, mask=batch_mask)

    pred_pointer = (input_pointer +
                    input_feat_stride * target +
                    input_batch_stride * batch_offset)
    input_pointer += (input_batch_stride * batch_offset[:, None] +
                      input_feat_stride * feat_offset[None, :])

    input = tl.load(input_pointer, mask=batch_mask[:, None] & feat_mask[None, :],
                    other=-float('inf')).to(tl.float32)
    pred = tl.load(pred_pointer, mask=batch_mask).to(tl.float32)

    # Log-sum-exp: loss = log(sum(exp(x - max))) + max - pred
    mx = tl.max(input, axis=1)
    input -= mx[:, None]
    loss = tl.log(tl.sum(tl.exp(input), axis=1)) - pred + mx

    if weighted:
        weight = tl.load(weight_pointer + target, mask=batch_mask).to(tl.float32)
        loss *= weight
        tl.store(sum_weights_pointer + batch_pid, tl.sum(weight))
    else:
        loss /= batch_dim

    tl.store(output_pointer + batch_pid, tl.sum(loss))
