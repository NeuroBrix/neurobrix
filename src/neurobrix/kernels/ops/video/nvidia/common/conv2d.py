# Conv2d - 2D Convolution
# Type: Triton Kernel (Complex)
# NeuroBrix - NVIDIA Common (All architectures)
# Reference: Implicit GEMM approach

import torch
import triton
import triton.language as tl
from typing import Tuple

from neurobrix.kernels.registry import register_kernel


@triton.jit
def _conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    # Input dims
    N, C_in, H_in, W_in,
    # Weight dims
    C_out, K_H, K_W,
    # Output dims
    H_out, W_out,
    # Strides
    stride_h, stride_w,
    # Padding
    pad_h, pad_w,
    # Input strides
    stride_xn, stride_xc, stride_xh, stride_xw,
    # Weight strides
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    # Output strides
    stride_on, stride_oc, stride_oh, stride_ow,
    # Block sizes
    BLOCK_M: tl.constexpr,  # Output channels
    BLOCK_N: tl.constexpr,  # Output spatial
    BLOCK_K: tl.constexpr,  # Reduction (C_in * K_H * K_W)
):
    """
    Conv2d using implicit GEMM.
    
    Reshapes convolution as: [N*H_out*W_out, C_in*K_H*K_W] @ [C_in*K_H*K_W, C_out]
    """
    pid = tl.program_id(0)
    
    # Compute output position
    pid_m = pid // tl.cdiv(N * H_out * W_out, BLOCK_N)  # C_out block
    pid_n = pid % tl.cdiv(N * H_out * W_out, BLOCK_N)   # Spatial block
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # C_out indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # Spatial indices
    
    # Decode spatial index to (n, h_out, w_out)
    n_idx = offs_n // (H_out * W_out)
    hw_idx = offs_n % (H_out * W_out)
    h_out_idx = hw_idx // W_out
    w_out_idx = hw_idx % W_out
    
    # Accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # Reduction over C_in * K_H * K_W
    K_total = C_in * K_H * K_W
    
    for k_start in range(0, K_total, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Decode k to (c_in, kh, kw)
        c_in_idx = offs_k // (K_H * K_W)
        khkw_idx = offs_k % (K_H * K_W)
        kh_idx = khkw_idx // K_W
        kw_idx = khkw_idx % K_W
        
        # Input position
        h_in_idx = h_out_idx[:, None] * stride_h - pad_h + kh_idx[None, :]
        w_in_idx = w_out_idx[:, None] * stride_w - pad_w + kw_idx[None, :]
        
        # Bounds check
        valid_h = (h_in_idx >= 0) & (h_in_idx < H_in)
        valid_w = (w_in_idx >= 0) & (w_in_idx < W_in)
        valid_k = offs_k[None, :] < K_total
        valid_n = offs_n[:, None] < N * H_out * W_out
        mask_x = valid_h & valid_w & valid_k & valid_n
        
        # Load input [BLOCK_N, BLOCK_K]
        x_offs = (n_idx[:, None] * stride_xn + 
                  c_in_idx[None, :] * stride_xc +
                  h_in_idx * stride_xh + 
                  w_in_idx * stride_xw)
        x = tl.load(x_ptr + x_offs, mask=mask_x, other=0.0)
        
        # Load weights [BLOCK_K, BLOCK_M]
        mask_w = (offs_k[:, None] < K_total) & (offs_m[None, :] < C_out)
        w_offs = (offs_m[None, :] * stride_wco +
                  c_in_idx[:, None] * stride_wci +
                  kh_idx[:, None] * stride_wkh +
                  kw_idx[:, None] * stride_wkw)
        w = tl.load(w_ptr + w_offs, mask=mask_w, other=0.0)
        
        # Accumulate
        acc += tl.dot(x, w)
    
    # Store output [BLOCK_N, BLOCK_M]
    mask_out = (offs_m[None, :] < C_out) & (offs_n[:, None] < N * H_out * W_out)
    out_offs = (n_idx[:, None] * stride_on +
                offs_m[None, :] * stride_oc +
                h_out_idx[:, None] * stride_oh +
                w_out_idx[:, None] * stride_ow)
    tl.store(out_ptr + out_offs, acc, mask=mask_out)


def _conv2d_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """Conv2d using Triton."""
    assert x.is_cuda and weight.is_cuda
    
    N, C_in, H_in, W_in = x.shape
    C_out, C_in_w, K_H, K_W = weight.shape
    assert C_in == C_in_w
    
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    
    H_out = (H_in + 2 * pad_h - K_H) // stride_h + 1
    W_out = (W_in + 2 * pad_w - K_W) // stride_w + 1
    
    x = x.contiguous()
    weight = weight.contiguous()
    
    out = torch.empty(N, C_out, H_out, W_out, dtype=x.dtype, device=x.device)
    
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32
    
    grid = (triton.cdiv(C_out, BLOCK_M) * triton.cdiv(N * H_out * W_out, BLOCK_N),)
    
    _conv2d_kernel[grid](
        x, weight, out,
        N, C_in, H_in, W_in,
        C_out, K_H, K_W,
        H_out, W_out,
        stride_h, stride_w,
        pad_h, pad_w,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    
    if bias is not None:
        out += bias.view(1, -1, 1, 1)
    
    return out

@triton.jit
def _conv2d_dilated_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C_in, H_in, W_in,
    C_out, K_H, K_W,
    H_out, W_out,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_CO: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """Conv2d with dilation support."""
    pid = tl.program_id(0)
    n_idx = tl.program_id(1)

    num_hw = H_out * W_out
    num_co_blocks = tl.cdiv(C_out, BLOCK_CO)

    co_block = pid // tl.cdiv(num_hw, BLOCK_HW)
    hw_block = pid % tl.cdiv(num_hw, BLOCK_HW)

    co_offs = co_block * BLOCK_CO + tl.arange(0, BLOCK_CO)
    hw_offs = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)

    h_out = hw_offs // W_out
    w_out = hw_offs % W_out

    acc = tl.zeros([BLOCK_CO, BLOCK_HW], dtype=tl.float32)

    for ci in range(C_in):
        for kh in range(K_H):
            for kw in range(K_W):
                h_in = h_out * stride_h - pad_h + kh * dil_h
                w_in = w_out * stride_w - pad_w + kw * dil_w

                valid = (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in)
                valid = valid & (hw_offs < num_hw)

                x_offs = n_idx * stride_xn + ci * stride_xc + h_in * stride_xh + w_in * stride_xw
                x_val = tl.load(x_ptr + x_offs, mask=valid, other=0.0)

                w_offs = co_offs[:, None] * stride_wco + ci * stride_wci + kh * stride_wkh + kw * stride_wkw
                w_mask = co_offs[:, None] < C_out
                w_val = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

                acc += w_val * x_val[None, :]

    out_offs = n_idx * stride_on + co_offs[:, None] * stride_oc + h_out[None, :] * stride_oh + w_out[None, :] * stride_ow
    out_mask = (co_offs[:, None] < C_out) & (hw_offs[None, :] < num_hw)
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


def _conv2d_dilated_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
) -> torch.Tensor:
    """Conv2d with dilation using Triton."""
    assert x.is_cuda and weight.is_cuda

    N, C_in, H_in, W_in = x.shape
    C_out, C_in_w, K_H, K_W = weight.shape
    assert C_in == C_in_w

    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    H_out = (H_in + 2 * pad_h - dil_h * (K_H - 1) - 1) // stride_h + 1
    W_out = (W_in + 2 * pad_w - dil_w * (K_W - 1) - 1) // stride_w + 1

    x = x.contiguous()
    weight = weight.contiguous()

    out = torch.empty(N, C_out, H_out, W_out, dtype=x.dtype, device=x.device)

    BLOCK_CO = 16
    BLOCK_HW = 64

    num_hw = H_out * W_out
    grid = (triton.cdiv(C_out, BLOCK_CO) * triton.cdiv(num_hw, BLOCK_HW), N)

    with torch.cuda.device(x.device):
        _conv2d_dilated_kernel[grid](
            x, weight, out,
            N, C_in, H_in, W_in,
            C_out, K_H, K_W,
            H_out, W_out,
            stride_h, stride_w,
            pad_h, pad_w,
            dil_h, dil_w,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            BLOCK_CO=BLOCK_CO, BLOCK_HW=BLOCK_HW,
            num_warps=4,
        )

    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)

    return out


@register_kernel(family="video", vendor="nvidia", tier="common", op_name="conv2d")
def conv2d_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    groups: int = 1,
) -> torch.Tensor:
    """Conv2d kernel - pure Triton."""
    # For groups > 1, split and process each group
    if groups > 1:
        C_in = x.shape[1]
        C_out = weight.shape[0]
        group_in = C_in // groups
        group_out = C_out // groups

        outputs = []
        for g in range(groups):
            x_g = x[:, g * group_in:(g + 1) * group_in, :, :]
            w_g = weight[g * group_out:(g + 1) * group_out, :, :, :]
            b_g = bias[g * group_out:(g + 1) * group_out] if bias is not None else None

            if dilation != (1, 1):
                out_g = _conv2d_dilated_impl(x_g, w_g, b_g, stride, padding, dilation)
            else:
                out_g = _conv2d_impl(x_g, w_g, b_g, stride, padding)
            outputs.append(out_g)

        # Concat along channel dim
        return _concat_groups(outputs)

    if dilation != (1, 1):
        return _conv2d_dilated_impl(x, weight, bias, stride, padding, dilation)
    return _conv2d_impl(x, weight, bias, stride, padding)


def _concat_groups(tensors):
    """Concatenate tensors along dim=1 using Triton copy."""
    if len(tensors) == 1:
        return tensors[0]

    device = tensors[0].device
    dtype = tensors[0].dtype

    # Calculate output shape
    out_shape = list(tensors[0].shape)
    out_shape[1] = sum(t.shape[1] for t in tensors)

    output = torch.empty(out_shape, dtype=dtype, device=device)

    # Copy each tensor
    offset = 0
    for t in tensors:
        n_channels = t.shape[1]
        output[:, offset:offset + n_channels, :, :] = t
        offset += n_channels

    return output


@register_kernel(family="video", vendor="nvidia", tier="common", op_name="convolution")
def convolution_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    groups: int = 1,
) -> torch.Tensor:
    """General convolution kernel - pure Triton."""
    return conv2d_kernel(x, weight, bias, stride, padding, dilation, groups)
