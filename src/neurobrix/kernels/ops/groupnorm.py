"""Group normalization — pure @triton.jit kernel.

Extracted from FlagGems (FlagOpen/FlagGems) groupnorm.py.
Forward only (inference). Stripped FlagGems-specific decorators.
"""

import triton
import triton.language as tl


@triton.jit
def group_norm_forward_kernel(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    group_size,
    C,
    HW,
    num_groups,
    eps,
    scale_by_weight: tl.constexpr,
    add_bias: tl.constexpr,
    BLOCK_GROUP_SIZE: tl.constexpr,
    BLOCK_HW_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    group = pid % num_groups
    num_elements = group_size * HW
    group_offset = tl.arange(0, BLOCK_GROUP_SIZE)
    hw_offset = tl.arange(0, BLOCK_HW_SIZE)

    wb_offset = group * group_size + group_offset
    wb_mask = wb_offset < C

    xy_offset = pid * num_elements + group_offset[:, None] * HW + hw_offset[None, :]
    xy_mask = (group_offset[:, None] < group_size) & (hw_offset[None, :] < HW)

    X_ptr = X + xy_offset
    Y_ptr = Y + xy_offset

    X_val = tl.load(X_ptr, mask=xy_mask, other=0.0).to(tl.float32)
    mean = tl.sum(X_val) / num_elements
    x = tl.where(xy_mask, X_val - mean, 0.0)

    var = tl.sum(x * x) / num_elements
    rstd = 1.0 / tl.sqrt(var + eps)
    x_hat = x * rstd

    if scale_by_weight:
        weight = tl.load(W + wb_offset, mask=wb_mask, other=1.0)[:, None]
        x_hat = x_hat * weight
    if add_bias:
        bias = tl.load(B + wb_offset, mask=wb_mask, other=0.0)[:, None]
        x_hat = x_hat + bias

    tl.store(Y_ptr, x_hat, mask=xy_mask)
    tl.store(Mean + pid, mean)
    tl.store(Rstd + pid, rstd)
