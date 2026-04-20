"""Matrix multiplication — pure @triton.jit kernel.

Ported from Triton official tutorial (BSD license).
Handles mm (2D), addmm via bias parameter.
bmm handled in wrappers via batch loop.
"""

import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    IEEE_PRECISION: tl.constexpr = False,
    PROMOTE_B: tl.constexpr = False,
):
    """C = A @ B where A is [M, K], B is [K, N], C is [M, N].

    Accumulates in fp32 for numerical stability.
    Output dtype determined by output pointer's dtype.

    IEEE_PRECISION=True forces `tl.dot(input_precision="ieee")` — required
    when fp32 inputs carry magnitudes > fp16_max on pre-Ampere GPUs,
    because `tl.dot` otherwise lowers through fp16 HMMA tensor cores which
    saturate the inputs to fp16 before the multiply. Set by the wrapper
    when `not _NBX_HAS_NATIVE_BF16` and inputs were promoted to fp32.

    PROMOTE_B=True casts the b tile to a's dtype after load and before
    tl.dot. Triton's type checker rejects `tl.dot(fp32, fp16)` at compile
    time; the cast is the cheapest way to bridge the mismatch — fused
    with the load, register-level, no heap allocation. Set by the wrapper
    when the activation was upcast fp16→fp32 (step 2) but the weight
    was left fp16 in memory (to save VRAM). The bit-exact fp32 promotion
    of a fp16 tile is free numerically (fp16 values are a subset of
    fp32); the accumulator is fp32 so the final dot product is identical
    to the path that widens the full weight pre-kernel.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        if PROMOTE_B:
            b = b.to(a.dtype)
        if IEEE_PRECISION:
            accumulator += tl.dot(a, b, input_precision="ieee")
        else:
            accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Write back — Triton auto-converts fp32 accum to output ptr dtype
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def addmm_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    alpha, beta,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    IEEE_PRECISION: tl.constexpr = False,
    PROMOTE_B: tl.constexpr = False,
):
    """C = beta * bias + alpha * (A @ B) where bias is [N].

    PROMOTE_B: see matmul_kernel docstring. Same in-kernel fp16→fp32
    tile cast; enables the wrapper to keep fp16 weights fp16 in memory
    while still running tl.dot with matched dtypes.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        if PROMOTE_B:
            b = b.to(a.dtype)
        if IEEE_PRECISION:
            accumulator += tl.dot(a, b, input_precision="ieee")
        else:
            accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias: C = alpha * matmul + beta * bias
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias_mask = offs_cn < N
    bias = tl.load(bias_ptr + offs_cn, mask=bias_mask)
    accumulator = alpha * accumulator + beta * bias[None, :]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
