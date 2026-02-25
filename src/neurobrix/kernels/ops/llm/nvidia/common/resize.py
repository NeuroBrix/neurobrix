"""
Resize kernel - Pure Triton
ATen: aten::upsample_nearest2d, aten::upsample_bilinear2d
"""
import torch
import triton
import triton.language as tl
from neurobrix.kernels.registry import register_kernel


@triton.jit
def _resize_nearest_kernel(
    x_ptr, out_ptr,
    N, C, H_in, W_in,
    H_out, W_out,
    scale_h, scale_w,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_SIZE: tl.constexpr
):
    """Nearest neighbor resize - pure Triton."""
    pid = tl.program_id(0)
    n_idx = tl.program_id(1)
    c_idx = tl.program_id(2)

    # Each block handles BLOCK_SIZE output pixels
    hw_start = pid * BLOCK_SIZE
    hw_offs = hw_start + tl.arange(0, BLOCK_SIZE)
    mask = hw_offs < H_out * W_out

    h_out = hw_offs // W_out
    w_out = hw_offs % W_out

    # Map to input coordinates (nearest)
    h_in = (h_out.to(tl.float32) / scale_h).to(tl.int32)
    w_in = (w_out.to(tl.float32) / scale_w).to(tl.int32)

    # Clamp
    h_in = tl.minimum(tl.maximum(h_in, 0), H_in - 1)
    w_in = tl.minimum(tl.maximum(w_in, 0), W_in - 1)

    # Load from input
    x_offs = n_idx * stride_xn + c_idx * stride_xc + h_in * stride_xh + w_in * stride_xw
    val = tl.load(x_ptr + x_offs, mask=mask, other=0.0)

    # Store to output
    out_offs = n_idx * stride_on + c_idx * stride_oc + h_out * stride_oh + w_out * stride_ow
    tl.store(out_ptr + out_offs, val, mask=mask)


@triton.jit
def _resize_bilinear_kernel(
    x_ptr, out_ptr,
    N, C, H_in, W_in,
    H_out, W_out,
    scale_h, scale_w,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_SIZE: tl.constexpr
):
    """Bilinear resize - pure Triton."""
    pid = tl.program_id(0)
    n_idx = tl.program_id(1)
    c_idx = tl.program_id(2)

    hw_start = pid * BLOCK_SIZE
    hw_offs = hw_start + tl.arange(0, BLOCK_SIZE)
    mask = hw_offs < H_out * W_out

    h_out = hw_offs // W_out
    w_out = hw_offs % W_out

    # Map to input coordinates
    h_in_f = (h_out.to(tl.float32) + 0.5) / scale_h - 0.5
    w_in_f = (w_out.to(tl.float32) + 0.5) / scale_w - 0.5

    # Get integer and fractional parts
    h0 = tl.floor(h_in_f).to(tl.int32)
    w0 = tl.floor(w_in_f).to(tl.int32)
    h1 = h0 + 1
    w1 = w0 + 1

    # Fractional parts
    fh = h_in_f - h0.to(tl.float32)
    fw = w_in_f - w0.to(tl.float32)

    # Clamp coordinates
    h0 = tl.minimum(tl.maximum(h0, 0), H_in - 1)
    h1 = tl.minimum(tl.maximum(h1, 0), H_in - 1)
    w0 = tl.minimum(tl.maximum(w0, 0), W_in - 1)
    w1 = tl.minimum(tl.maximum(w1, 0), W_in - 1)

    # Load 4 corners
    base = n_idx * stride_xn + c_idx * stride_xc
    v00 = tl.load(x_ptr + base + h0 * stride_xh + w0 * stride_xw, mask=mask, other=0.0)
    v01 = tl.load(x_ptr + base + h0 * stride_xh + w1 * stride_xw, mask=mask, other=0.0)
    v10 = tl.load(x_ptr + base + h1 * stride_xh + w0 * stride_xw, mask=mask, other=0.0)
    v11 = tl.load(x_ptr + base + h1 * stride_xh + w1 * stride_xw, mask=mask, other=0.0)

    # Bilinear interpolation
    val = (v00 * (1.0 - fh) * (1.0 - fw) +
           v01 * (1.0 - fh) * fw +
           v10 * fh * (1.0 - fw) +
           v11 * fh * fw)

    # Store
    out_offs = n_idx * stride_on + c_idx * stride_oc + h_out * stride_oh + w_out * stride_ow
    tl.store(out_ptr + out_offs, val, mask=mask)


def _resize_impl(x: torch.Tensor, size: list, mode: str = "nearest") -> torch.Tensor:
    """Resize implementation using Triton."""
    assert x.is_cuda
    x = x.contiguous()

    # Assume NCHW format
    N, C, H_in, W_in = x.shape
    H_out, W_out = size

    scale_h = float(H_out) / float(H_in)
    scale_w = float(W_out) / float(W_in)

    out = torch.empty(N, C, H_out, W_out, dtype=x.dtype, device=x.device)

    BLOCK_SIZE = 256
    num_hw = H_out * W_out
    grid = (triton.cdiv(num_hw, BLOCK_SIZE), N, C)

    with torch.cuda.device(x.device):
        if mode == "nearest":
            _resize_nearest_kernel[grid](
                x, out,
                N, C, H_in, W_in,
                H_out, W_out,
                scale_h, scale_w,
                x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=4,
            )
        else:  # bilinear
            _resize_bilinear_kernel[grid](
                x, out,
                N, C, H_in, W_in,
                H_out, W_out,
                scale_h, scale_w,
                x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=4,
            )

    return out


@register_kernel(family="llm", vendor="nvidia", tier="common", op_name="resize")
def resize_kernel(x: torch.Tensor, size: list = None, mode: str = "nearest") -> torch.Tensor:
    """Resize tensor - pure Triton."""
    if size is None:
        return x
    return _resize_impl(x, size, mode)

