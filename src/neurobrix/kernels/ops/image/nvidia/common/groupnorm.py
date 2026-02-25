# Group Normalization - Pure Triton
# Source: Liger-Kernel (Apache 2.0)
# NeuroBrix - NVIDIA Common

import torch
import triton
import triton.language as tl

# Helper for rsqrt
@triton.jit
def _rsqrt(x):
    return tl.math.rsqrt(x)

@triton.jit
def _group_norm_forward_kernel(
    Y_ptr,  # pointer to output
    Y_row_stride, Y_col_stride,
    X_ptr,  # pointer to input
    X_row_stride, X_col_stride,
    Mean_ptr,  # pointer to mean
    Mean_row_stride, Mean_col_stride,
    RSTD_ptr,  # pointer to rstd
    RSTD_row_stride, RSTD_col_stride,
    W_ptr,  # pointer to W
    B_ptr,  # pointer to B
    hidden_size,  # hidden size of X (flattened spatial)
    channels_per_group,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)

    X_ptr += batch_idx * X_row_stride + group_idx * X_col_stride
    Y_ptr += batch_idx * Y_row_stride + group_idx * Y_col_stride

    block_range = tl.arange(0, BLOCK_SIZE)

    # Compute mean and variance
    s = 0.0
    squared_sum = 0.0
    for i in range(0, hidden_size, BLOCK_SIZE):
        offsets = i + block_range
        mask = offsets < hidden_size
        val = tl.load(X_ptr + offsets, mask=mask, other=0.0)
        s += tl.sum(val)
        squared_sum += tl.sum(val * val)

    m = s / hidden_size
    variance = (squared_sum / hidden_size) - (m * m)
    rstd = _rsqrt(variance + eps)

    # Normalize
    # We iterate over channels within this group
    # Structure: [N, G, C_per_G, HW] flattened to [N, G, HiddenSize] where HiddenSize = C_per_G * HW?
    # Liger assumes X is [N, G, HiddenSize].
    # But GroupNorm is usually [N, C, H, W].
    # So HiddenSize here is (C//G) * H * W.
    
    # Wait, Liger's logic:
    # for channel_idx in range(group_idx * channels_per_group, ...)
    # It assumes W and B are per channel.
    # And it loops over channels.
    
    # If hidden_size is the total size of the group's data, how do we split by channel?
    # We need to know spatial size (HW).
    # Liger code:
    # hidden_size_per_channel = hidden_size // channels_per_group
    
    hidden_size_per_channel = hidden_size // channels_per_group
    
    # Iterate over channels in this group
    start_c = group_idx * channels_per_group
    end_c = start_c + channels_per_group
    
    curr_x_ptr = X_ptr
    curr_y_ptr = Y_ptr
    
    for c in range(start_c, end_c):
        w = tl.load(W_ptr + c) if W_ptr is not None else 1.0
        b = tl.load(B_ptr + c) if B_ptr is not None else 0.0
        
        for i in range(0, hidden_size_per_channel, BLOCK_SIZE):
            offsets = i + block_range
            mask = offsets < hidden_size_per_channel
            val = tl.load(curr_x_ptr + offsets, mask=mask, other=0.0)
            norm = (val - m) * rstd * w + b
            tl.store(curr_y_ptr + offsets, norm, mask=mask)
            
        curr_x_ptr += hidden_size_per_channel
        curr_y_ptr += hidden_size_per_channel

    # Store stats
    tl.store(Mean_ptr + batch_idx * Mean_row_stride + group_idx * Mean_col_stride, m)
    tl.store(RSTD_ptr + batch_idx * RSTD_row_stride + group_idx * RSTD_col_stride, rstd)
