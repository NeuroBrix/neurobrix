"""Average pooling 2D — pure @triton.jit kernel.

Extracted from FlagGems avg_pool2d.py. Forward only (inference).
2D blocked approach with proper padding, dilation, count_include_pad support.
"""

import triton
import triton.language as tl


@triton.jit
def avg_pool2d_forward_kernel(
    input_ptr,
    output_ptr,
    in_stride_n, in_stride_c, in_stride_h, in_stride_w,
    in_c, in_h, in_w,
    out_h, out_w,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    divisor_override,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    sum_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    count_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int32)

    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    for kh in range(0, kernel_h):
        for kw in range(0, kernel_w):
            h_in = h_out_offsets[:, None] * stride_h - padding_h + kh * dilation_h
            w_in = w_out_offsets[None, :] * stride_w - padding_w + kw * dilation_w
            in_mask = (h_in >= 0) & (h_in < in_h) & (w_in >= 0) & (w_in < in_w)

            input_offset = h_in * in_stride_h + w_in * in_stride_w
            current_val = tl.load(input_base_ptr + input_offset, mask=in_mask, other=0.0)

            sum_acc += tl.where(in_mask, current_val, 0.0)
            count_acc += in_mask.to(tl.int32)

    if divisor_override != 0:
        divisor = tl.full((BLOCK_H, BLOCK_W), divisor_override, dtype=tl.float32)
    elif COUNT_INCLUDE_PAD:
        divisor = tl.full((BLOCK_H, BLOCK_W), kernel_h * kernel_w, dtype=tl.float32)
    else:
        divisor = count_acc.to(tl.float32)

    output_vals = tl.where(divisor != 0, sum_acc / divisor, 0.0)

    out_base_ptr = output_ptr + pid_nc * out_h * out_w
    output_block_ptr = (
        out_base_ptr + h_out_offsets[:, None] * out_w + w_out_offsets[None, :]
    )
    out_mask = (h_out_offsets[:, None] < out_h) & (w_out_offsets[None, :] < out_w)
    tl.store(output_block_ptr, output_vals.to(output_ptr.type.element_ty), mask=out_mask)
