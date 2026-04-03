"""Conv1D forward kernel — adapted from FlagGems conv2d forward kernel for 1D.

Pure Triton. ZERO PyTorch imports.

The FlagGems conv1d reference simply delegates to conv2d via unsqueeze.
This kernel is a direct 1D specialization of the conv2d_forward_kernel,
operating on [N, C_in, L_in] input and [C_out, C_in/groups, K] weight.
"""

import triton
import triton.language as tl


@triton.jit
def conv1d_forward_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
    in_n,
    input_length,
    out_c,
    out_length,
    input_n_stride,
    input_c_stride,
    input_l_stride,
    weight_n_stride,
    weight_c_stride,
    weight_l_stride,
    output_n_stride,
    output_c_stride,
    output_l_stride,
    weight_c: tl.constexpr,
    weight_k: tl.constexpr,
    stride_l: tl.constexpr,
    padding_l: tl.constexpr,
    dilation_l: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_NI_LO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    """Conv1D forward: [N, C_in, L] * [C_out, C_in/g, K] -> [N, C_out, L_out]."""
    pid_ni_lo = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_group = tl.program_id(2)

    # Decompose flattened (N * L_out) index
    ni_lo_offset = pid_ni_lo * BLOCK_NI_LO + tl.arange(0, BLOCK_NI_LO)
    in_n_point = ni_lo_offset // out_length
    out_l_point = ni_lo_offset % out_length

    out_per_group_c = out_c // groups
    output_c_offset = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)

    # Advance input pointer to correct batch and group
    input_pointer += (input_n_stride * in_n_point
                      + input_c_stride * pid_group * weight_c)[:, None]
    # Advance weight pointer to correct output channel and group
    weight_pointer += (weight_n_stride * output_c_offset
                       + weight_n_stride * pid_group * out_per_group_c)[None, :]

    accum = tl.zeros((BLOCK_NI_LO, BLOCK_CO), dtype=tl.float32)
    BLOCK_CI_COUNT = (weight_c + BLOCK_CI - 1) // BLOCK_CI

    for lc in range(weight_k * BLOCK_CI_COUNT):
        c = (lc % BLOCK_CI_COUNT) * BLOCK_CI
        k = lc // BLOCK_CI_COUNT

        input_c_offset = c + tl.arange(0, BLOCK_CI)
        input_l_offset = k * dilation_l - padding_l + stride_l * out_l_point

        curr_input_pointer = (
            input_pointer
            + (input_c_stride * input_c_offset)[None, :]
            + (input_l_stride * input_l_offset)[:, None]
        )
        curr_weight_pointer = (
            weight_pointer
            + (weight_c_stride * input_c_offset)[:, None]
            + (weight_l_stride * k)
        )

        input_mask = (
            (in_n_point < in_n)[:, None]
            & (input_c_offset < weight_c)[None, :]
            & (0 <= input_l_offset)[:, None]
            & (input_l_offset < input_length)[:, None]
        )
        weight_mask = (
            (input_c_offset < weight_c)[:, None]
            & (output_c_offset < out_per_group_c)[None, :]
        )

        input_block = tl.load(curr_input_pointer, mask=input_mask, other=0.0)
        weight_block = tl.load(curr_weight_pointer, mask=weight_mask, other=0.0)
        accum += tl.dot(input_block, weight_block, allow_tf32=False)

    # Add bias
    bias_pointer += (pid_group * out_per_group_c + output_c_offset)[None, :]
    mask_bias = (output_c_offset < out_per_group_c)[None, :]
    bias = tl.load(bias_pointer, mask=mask_bias, other=0.0).to(tl.float32)
    accum += bias

    # Store output
    output_pointer += (
        (output_n_stride * in_n_point)[:, None]
        + (output_c_stride * (pid_group * out_per_group_c + output_c_offset))[None, :]
        + (output_l_stride * out_l_point)[:, None]
    )
    output_mask = (
        (in_n_point < in_n)[:, None]
        & (output_c_offset < out_per_group_c)[None, :]
        & (out_l_point < out_length)[:, None]
    )
    tl.store(output_pointer, accum, mask=output_mask)
