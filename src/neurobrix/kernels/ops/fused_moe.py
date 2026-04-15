"""Fused MoE kernels — pure @triton.jit, zero torch.

fused_moe_kernel: Grouped GEMM for all experts in one launch.
  Expert weights accessed via absolute pointer table (int64 per expert).
  Same approach works for scattered arena allocations — no offset arithmetic.

silu_and_mul_kernel: Fused SwiGLU activation.
"""

import triton
import triton.language as tl


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,                          # activations [M, K]
    expert_ptrs_ptr,                # [E] int64 — absolute weight pointers
    c_ptr,                          # output [num_tokens_post_padded, N]
    topk_weights_ptr,               # routing scores [M * top_k]
    sorted_token_ids_ptr,           # sorted token indices
    expert_ids_ptr,                 # expert id per BLOCK_M group
    num_tokens_post_padded_ptr,     # [1] total tokens after padding
    # Matrix dimensions
    N,                              # output feature dim
    K,                              # input feature dim (reduction)
    EM,                             # total sorted entries (with padding)
    num_valid_tokens,               # M * top_k (before padding)
    # Strides (shared across all experts — same shape)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    TOPK_DIVIDE: tl.constexpr = True,
):
    """Grouped GEMM with absolute expert pointer table.

    Each BLOCK_M block of sorted tokens shares one expert. The kernel loads
    that expert's absolute weight pointer from the table and uses it directly.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # Load sorted token IDs for this block
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    # Load expert id → load absolute weight pointer from table
    off_experts = tl.load(expert_ids_ptr + pid_m)
    expert_ptr_int = tl.load(expert_ptrs_ptr + off_experts)
    # Bitcast int64 to pointer — reinterpret bits as address
    b_base = tl.cast(expert_ptr_int, tl.pointer_type(compute_type), bitcast=True)

    # INT64 offsets for pointer arithmetic within the expert weight matrix
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

    if TOPK_DIVIDE:
        a_idx = offs_token[:, None] // top_k
    else:
        a_idx = offs_token[:, None]
    a_ptrs = a_ptr + (a_idx * stride_am + offs_k[None, :] * stride_ak)

    # B pointers: absolute expert pointer + in-matrix offsets
    b_ptrs = b_base + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
            other=0.0,
        )
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token,
            mask=token_mask,
            other=0,
        )
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)).to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def silu_and_mul_kernel(
    input_ptr, output_ptr,
    M, N,
    stride_im, stride_in,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused SwiGLU: output = silu(input[:, :N]) * input[:, N:2*N]."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    gate_ptrs = input_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
    up_ptrs = input_ptr + offs_m[:, None] * stride_im + (offs_n[None, :] + N) * stride_in

    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)

    silu_gate = gate * tl.sigmoid(gate)
    result = silu_gate * up

    out_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, result, mask=mask)


@triton.jit
def silu_mul_split_kernel(
    gate_ptr, up_ptr, output_ptr,
    M, N,
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused SwiGLU with split gate/up tensors: output = silu(gate) * up.

    Variant of silu_and_mul_kernel for the common case where gate and up
    are produced as two separate tensors (no concat needed).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    gate_ptrs = gate_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
    up_ptrs = up_ptr + offs_m[:, None] * stride_um + offs_n[None, :] * stride_un

    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)

    silu_gate = gate * tl.sigmoid(gate)
    result = silu_gate * up

    out_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, result, mask=mask)
