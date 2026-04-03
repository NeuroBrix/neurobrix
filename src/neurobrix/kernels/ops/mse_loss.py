"""MSE Loss kernels — extracted from FlagGems.

Pure Triton. ZERO PyTorch imports.

Two-pass reduction:
  kernel_1: Per-block partial (input - target)^2, reduced by mean or sum.
  kernel_2: Final reduction across blocks.

Reduction modes: 1 = mean, 2 = sum.
For reduction=0 (none), use a simple pointwise kernel instead.
"""

import triton
import triton.language as tl

@triton.jit
def mse_loss_partial_kernel(
    inp,
    target,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
    reduction: tl.constexpr,
):
    """Pass 1: per-block squared difference, reduced to partial sums.

    For reduction==1 (mean): each block stores sum((inp-target)^2) / M.
    For reduction==2 (sum):  each block stores sum((inp-target)^2).
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M

    inp_val = tl.load(inp + offset, mask=mask, other=0).to(tl.float32)
    target_val = tl.load(target + offset, mask=mask, other=0).to(tl.float32)
    sub = inp_val - target_val
    pow_val = sub * sub

    # reduction == 1: mean, reduction == 2: sum
    if reduction == 1:
        sum_val = tl.sum(pow_val) / M
    else:
        sum_val = tl.sum(pow_val)

    tl.store(mid + pid, sum_val)

@triton.jit
def mse_loss_reduce_kernel(
    mid,
    out,
    mid_size,
    BLOCK_MID: tl.constexpr,
):
    """Pass 2: sum all partial results into a single scalar."""
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    mid_val = tl.load(mid + offset, mask=mask, other=0).to(tl.float32)
    sum_val = tl.sum(mid_val)
    tl.store(out, sum_val)

@triton.jit
def mse_loss_none_kernel(
    inp_ptr,
    target_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Pointwise (no reduction): output[i] = (input[i] - target[i])^2."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(inp_ptr + offset, mask=mask, other=0).to(tl.float32)
    y = tl.load(target_ptr + offset, mask=mask, other=0).to(tl.float32)

    diff = x - y
    result = diff * diff
    tl.store(output_ptr + offset, result, mask=mask)
