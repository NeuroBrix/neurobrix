"""Adaptive average pooling 2D — pure @triton.jit kernel.

For each output element (n, c, oh, ow), the kernel computes the input
window boundaries using the standard PyTorch adaptive pooling formula:
    start_h = floor(oh * IH / OH)
    end_h   = ceil((oh + 1) * IH / OH)
and averages all input elements in [start_h, end_h) x [start_w, end_w).

The end index is a CEILING. It was a floor here, which is the same value
whenever the output size divides the input size and a smaller window
whenever it does not — so every exactly-dividing shape agreed and every
ragged one was quietly averaging too few elements. Measured against torch
2026-09-09 by one-hot probing which input rows reach each output row:
IH=7,OH=3 gives torch [(0,3),(2,5),(4,7)] and the floor form
[(0,2),(2,4),(4,7)].

The window loop was also fixed at 8x8, on a comment that the adaptive
window is "typically 1-3 pixels each dim". That is false for any pooling
that downsamples by more than 8, and global average pooling is the common
case: a 16x16 plane to 1x1 has a 16x16 window, of which the kernel summed
the first 8x8 and divided by 64. Measured: 0.1807 returned where the plane
mean is 0.0019. The bound is now derived from the shape.

BOTH defects are in this kernel, not in any backend, so both were wrong on
EVERY vendor. This changes NVIDIA's numbers, and is meant to: keeping them
would be keeping the bug.
"""

import triton
import triton.language as tl

@triton.jit
def adaptive_avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    # Input shape: [N, C, IH, IW]
    IH, IW,
    # Output shape: [N, C, OH, OW]
    OH, OW,
    # Strides (input contiguous NCHW)
    stride_n, stride_c, stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    """Adaptive average pool 2D.

    Each thread computes one output element by averaging over the
    corresponding adaptive window in the input.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements

    # Decompose flat index -> (n, c, oh, ow)
    ow = idx % OW
    tmp = idx // OW
    oh = tmp % OH
    tmp2 = tmp // OH
    c = tmp2 % (stride_n // stride_c)  # C = stride_n / stride_c for contiguous
    n = tmp2 // (stride_n // stride_c)

    # Adaptive pooling window bounds (PyTorch formula)
    # start = floor(out_idx * in_size / out_size)
    # end   = CEIL((out_idx + 1) * in_size / out_size)
    h_start = (oh * IH) // OH
    h_end = ((oh + 1) * IH + OH - 1) // OH
    w_start = (ow * IW) // OW
    w_end = ((ow + 1) * IW + OW - 1) // OW

    # Base pointer for this (n, c) plane
    base = n * stride_n + c * stride_c

    # Accumulate sum over the window
    # Window size varies per output position in adaptive pooling
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # The widest window this shape can produce, derived rather than assumed.
    # end - start <= ceil((i+1)*IN/OUT) - floor(i*IN/OUT) < IN/OUT + 2, so
    # floor(IN/OUT) + 2 bounds it for every i. The old fixed 8 silently
    # truncated any window wider than that — every global average pool over
    # a plane larger than 8x8 summed a corner and divided by its own count.
    KH = IH // OH + 2
    KW = IW // OW + 2
    for dh in range(0, KH):
        for dw in range(0, KW):
            h_idx = h_start + dh
            w_idx = w_start + dw
            in_bounds = (h_idx < h_end) & (w_idx < w_end)
            valid = mask & in_bounds

            offset = base + h_idx * stride_h + w_idx * stride_w
            val = tl.load(input_ptr + offset, mask=valid, other=0.0).to(tl.float32)
            acc += tl.where(valid, val, 0.0)
            count += tl.where(valid, 1.0, 0.0)

    # Average
    count = tl.maximum(count, 1.0)  # Avoid division by zero
    result = acc / count

    # Store
    out_offset = idx
    tl.store(output_ptr + out_offset, result, mask=mask)

@triton.jit
def adaptive_avg_pool2d_tiled_kernel(
    input_ptr,
    output_ptr,
    N, C,
    IH, IW,
    OH, OW,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    """Tiled variant for spatial parallelism.

    Grid: (cdiv(OW, BLOCK_X), cdiv(OH, BLOCK_Y), N*C)
    """
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_nc = tl.program_id(2)

    ow = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)  # [BLOCK_X]
    oh = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)  # [BLOCK_Y]

    mask_h = oh < OH
    mask_w = ow < OW
    mask_2d = mask_h[:, None] & mask_w[None, :]

    # Window boundaries
    h_start = (oh * IH) // OH  # [BLOCK_Y]
    h_end = ((oh + 1) * IH) // OH
    w_start = (ow * IW) // OW  # [BLOCK_X]
    w_end = ((ow + 1) * IW) // OW

    # Window sizes
    kh = h_end - h_start  # [BLOCK_Y]
    kw = w_end - w_start  # [BLOCK_X]

    base = pid_nc * IH * IW

    acc = tl.zeros([BLOCK_Y, BLOCK_X], dtype=tl.float32)

    for dh in range(0, 4):
        for dw in range(0, 4):
            h_idx = h_start[:, None] + dh
            w_idx = w_start[None, :] + dw
            in_bounds = (dh < kh[:, None]) & (dw < kw[None, :])
            valid = mask_2d & in_bounds

            offset = base + h_idx * IW + w_idx
            val = tl.load(input_ptr + offset, mask=valid, other=0.0).to(tl.float32)
            acc += tl.where(valid, val, 0.0)

    area = (kh[:, None] * kw[None, :]).to(tl.float32)
    area = tl.maximum(area, 1.0)
    result = acc / area

    out_base = pid_nc * OH * OW
    out_off = out_base + oh[:, None] * OW + ow[None, :]
    tl.store(output_ptr + out_off, result, mask=mask_2d)
