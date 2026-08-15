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
def add_scalar_dev_kernel(
    x_ptr, s_ptr, output_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    """out = x + (s * alpha) where s is a 0-d DEVICE tensor.

    Bit-exact mirror of add_scalar_kernel's `x + scalar` path: the
    host path computes `float(s) * alpha` in Python float64 then
    narrows to the f32 kernel argument; here the same math runs
    in-register (load -> f64 -> *alpha -> f32). Replaces the host
    .item() sync so the scalar stays device-resident — the replay
    device-scalar increment (Ming timestep class, 2026-08-15).

    Bit-exactness constraint: `alpha` binds as an f32 runtime arg, so
    the f64 multiply sees f32(alpha) where the host path sees the full
    f64 alpha — a 1-ulp double-rounding risk for alpha values not
    exactly representable in f32. Every ATen call site passes alpha=±1
    (exact); a non-representable alpha would need an f64-carried arg.
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    s = (tl.load(s_ptr).to(tl.float64) * alpha).to(tl.float32)
    x = tl.load(x_ptr + offset, mask=mask)
    tl.store(output_ptr + offset, x + s, mask=mask)


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
    # Cast to int64 — tensors >= 2^31 elements (e.g. Sana 4Kpx VAE
    # add::88 input 1x4096x4096x128 = 2^31) overflow int32 offset
    # arithmetic silently and corrupt `offset % feat_dim`, producing
    # garbage output. P-SANA-4KPX-RUNTIME 2026-05-07.
    offset = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    mask = offset < n_elements

    c = offset % feat_dim
    x = tl.load(x_ptr + offset, mask=mask)
    b = tl.load(bias_ptr + c, mask=mask)
    tl.store(output_ptr + offset, x + alpha * b, mask=mask)
