"""Conv2D forward — pure @triton.jit kernel.

From attorch (BobMcDear/attorch) conv_kernels.py.
Chosen over FlagGems for V100 compatibility (proper num_stages handling).
Im2col approach with tl.dot, fp32 accumulation, groups support.
"""

import triton
import triton.language as tl


@triton.jit
def conv2d_forward_kernel(
    input_pointer, weight_pointer, output_pointer,
    batch_dim, in_feat_dim, in_height, in_width,
    out_feat_dim, out_height, out_width,
    input_batch_stride, input_in_feat_stride, input_height_stride, input_width_stride,
    weight_out_feat_stride, weight_in_feat_stride, weight_height_stride, weight_width_stride,
    output_batch_stride, output_out_feat_stride, output_height_stride, output_width_stride,
    kernel_height: tl.constexpr, kernel_width: tl.constexpr,
    stride_height: tl.constexpr, stride_width: tl.constexpr,
    padding_height: tl.constexpr, padding_width: tl.constexpr,
    groups: tl.constexpr, fp16: tl.constexpr,
    BLOCK_SIZE_BHW: tl.constexpr, BLOCK_SIZE_INF: tl.constexpr,
    BLOCK_SIZE_OUTF: tl.constexpr,
):
    bhw_pid = tl.program_id(0)
    outf_pid = tl.program_id(1)
    group_pid = tl.program_id(2)

    in_group_dim = in_feat_dim // groups
    out_group_dim = out_feat_dim // groups

    bhw_offset = bhw_pid * BLOCK_SIZE_BHW + tl.arange(0, BLOCK_SIZE_BHW)
    bh_offset = bhw_offset // out_width
    batch_offset = bh_offset // out_height

    outf_offset = outf_pid * BLOCK_SIZE_OUTF + tl.arange(0, BLOCK_SIZE_OUTF)
    oh_offset = bh_offset % out_height
    ow_offset = bhw_offset % out_width

    input_pointer += (input_batch_stride * batch_offset +
                      input_in_feat_stride * group_pid * in_group_dim)[:, None]
    weight_pointer += (weight_out_feat_stride * outf_offset +
                       weight_out_feat_stride * group_pid * out_group_dim)[None, :]

    accum = tl.zeros((BLOCK_SIZE_BHW, BLOCK_SIZE_OUTF), dtype=tl.float32)

    for h in range(kernel_height):
        for w in range(kernel_width):
            for c in range(0, in_group_dim, BLOCK_SIZE_INF):
                inf_offset = c + tl.arange(0, BLOCK_SIZE_INF)
                ih_offset = h - padding_height + stride_height * oh_offset
                iw_offset = w - padding_width + stride_width * ow_offset

                curr_inp = (input_pointer +
                            (input_in_feat_stride * inf_offset)[None, :] +
                            (input_height_stride * ih_offset)[:, None] +
                            (input_width_stride * iw_offset)[:, None])
                curr_wt = (weight_pointer +
                           (weight_in_feat_stride * inf_offset)[:, None] +
                           (weight_height_stride * h) +
                           (weight_width_stride * w))

                inp_mask = ((batch_offset < batch_dim)[:, None] &
                            (inf_offset < in_group_dim)[None, :] &
                            (ih_offset >= 0)[:, None] &
                            (ih_offset < in_height)[:, None] &
                            (iw_offset >= 0)[:, None] &
                            (iw_offset < in_width)[:, None])
                wt_mask = ((inf_offset < in_group_dim)[:, None] &
                           (outf_offset < out_group_dim)[None, :])

                inp_block = tl.load(curr_inp, mask=inp_mask, other=0.0)
                wt_block = tl.load(curr_wt, mask=wt_mask, other=0.0)

                if fp16:
                    inp_block = inp_block.to(tl.float16)
                    wt_block = wt_block.to(tl.float16)

                accum += tl.dot(inp_block, wt_block)

    out_ptr = (output_pointer +
               (output_batch_stride * batch_offset)[:, None] +
               (output_out_feat_stride * (group_pid * out_group_dim + outf_offset))[None, :] +
               (output_height_stride * oh_offset)[:, None] +
               (output_width_stride * ow_offset)[:, None])
    out_mask = ((batch_offset < batch_dim)[:, None] &
                (outf_offset < out_group_dim)[None, :] &
                (oh_offset < out_height)[:, None] &
                (ow_offset < out_width)[:, None])

    tl.store(out_ptr, accum, mask=out_mask)
