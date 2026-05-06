"""Add — pure @triton.jit kernel (tensor + tensor, tensor + scalar)."""

import triton
import triton.language as tl

@triton.jit
def add_forward_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    """out = x + alpha * y (tensor + tensor)"""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)
    tl.store(output_ptr + offset, x + alpha * y, mask=mask)

@triton.jit
def add_scalar_kernel(
    x_ptr, output_ptr,
    n_elements,
    scalar,
    BLOCK_SIZE: tl.constexpr,
):
    """out = x + scalar"""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask)
    tl.store(output_ptr + offset, x + scalar, mask=mask)


@triton.jit
def add_bias_broadcast_kernel(
    x_ptr, bias_ptr, output_ptr,
    n_elements, feat_dim,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    """out[i] = x[i] + alpha * bias[i % feat_dim].

    Used by `add` wrapper when `y` is a 1D bias of size `feat_dim` that
    broadcasts against `x`'s last dim. Avoids materializing the 8 GiB
    contiguous broadcast view of bias on Sana 4Kpx VAE add::88
    (`mul::58::out_0 (1, 4096, 4096, 128) + bias (128)`). Reads bias
    directly via `offset % feat_dim` indexing — same effect as a
    stride-0 broadcast but compatible with the kernel's flat 1D
    addressing.
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    c = offset % feat_dim
    x = tl.load(x_ptr + offset, mask=mask)
    b = tl.load(bias_ptr + c, mask=mask)
    tl.store(output_ptr + offset, x + alpha * b, mask=mask)
