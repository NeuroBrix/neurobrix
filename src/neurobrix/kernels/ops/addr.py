"""Addr (outer product add) — pure @triton.jit kernel.

Computes: output = beta * input + alpha * (vec1 outer vec2)

Where vec1 is [M], vec2 is [N], input is [M, N] (broadcast-compatible),
and output is [M, N].

Extracted from FlagGems reference (addr.py).
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=['beta', 'alpha'])
def addr_kernel(
    input_ptr,
    vec1_ptr,
    vec2_ptr,
    output_ptr,
    beta,
    alpha,
    M, N,
    stride_input_m,
    stride_input_n,
    stride_vec1,
    stride_vec2,
    stride_output_m,
    stride_output_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Outer product add: out = beta * input + alpha * (vec1 outer vec2).

    Grid: (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_2d = mask_m[:, None] & mask_n[None, :]

    # Load vectors
    v1 = tl.load(vec1_ptr + offs_m * stride_vec1, mask=mask_m, other=0.0).to(tl.float32)
    v2 = tl.load(vec2_ptr + offs_n * stride_vec2, mask=mask_n, other=0.0).to(tl.float32)

    # Load input matrix
    input_ptrs = (
        input_ptr
        + offs_m[:, None] * stride_input_m
        + offs_n[None, :] * stride_input_n
    )
    inp = tl.load(input_ptrs, mask=mask_2d, other=0.0).to(tl.float32)

    # Compute: beta * input + alpha * outer(vec1, vec2)
    result = beta * inp + alpha * (v1[:, None] * v2[None, :])

    # Store
    output_ptrs = (
        output_ptr
        + offs_m[:, None] * stride_output_m
        + offs_n[None, :] * stride_output_n
    )
    tl.store(output_ptrs, result, mask=mask_2d)
