"""Fused MoE kernels — pure @triton.jit, zero torch.

Two kernels:
  fused_moe_kernel: Grouped GEMM for all experts in one launch.
    Each program handles one [BLOCK_M, BLOCK_N] tile. Tokens are sorted
    by expert via moe_align_block_size so each BLOCK_M group belongs to
    one expert. Weight matrix selected via expert offset table (zero-copy).

  silu_and_mul_kernel: Fused SwiGLU activation.
    Input [M, 2*N] → output [M, N]: silu(input[:, :N]) * input[:, N:]

Adapted from vLLM fused_moe.py (Apache-2.0), stripped of quantization.
Works with raw GPU pointers (NBXTensor.data_ptr()).

Memory: ZERO extra GPU memory for weight stacking.
  Expert weights stay in the arena. A small offset table ([E] int64 values)
  lets the kernel jump to each expert's weight matrix via pointer arithmetic.
"""

import triton
import triton.language as tl


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,                          # activations [M, K]
    weight_base_ptr,                # base pointer (any expert's data_ptr)
    expert_offsets_ptr,             # [E] element offsets from weight_base_ptr
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
    stride_am, stride_ak,           # A strides [M, K]
    stride_bk, stride_bn,          # B strides per expert (no expert stride)
    stride_cm, stride_cn,           # C strides
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
    """Grouped GEMM with indirect expert access.

    Each BLOCK_M block of sorted tokens shares one expert. The kernel loads
    the expert's weight base offset from expert_offsets_ptr, then does
    standard tiled matmul. Accumulates in fp32 for numerical stability.

    Zero extra GPU memory — expert weights stay in the arena.
    """
    # Map program id to tile (grouped ordering for L2 reuse)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Early exit for padding blocks
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # Load sorted token IDs for this block → indirect index into A
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    # Load expert ID → load element offset from table
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    expert_elem_offset = tl.load(expert_offsets_ptr + off_experts)

    # Build A pointers: indirect via sorted token IDs
    # TOPK_DIVIDE=True: A=hidden_states indexed by token → divide by top_k
    # TOPK_DIVIDE=False: A=activated indexed by flat routing index → use directly
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if TOPK_DIVIDE:
        a_idx = offs_token[:, None] // top_k
    else:
        a_idx = offs_token[:, None]
    a_ptrs = a_ptr + (a_idx * stride_am + offs_k[None, :] * stride_ak)

    # Build B pointers: base + expert element offset → per-expert weight matrix
    b_ptrs = (
        weight_base_ptr
        + expert_elem_offset
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    # Tiled GEMM loop — accumulate in fp32
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

    # Apply routing weights (multiply each token's output by its score)
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token,
            mask=token_mask,
            other=0,
        )
        accumulator = accumulator * moe_weight[:, None]

    # Cast to output dtype and write back
    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def silu_and_mul_kernel(
    input_ptr,
    output_ptr,
    M,
    N,                          # output N (input is 2*N)
    stride_im, stride_in,      # input strides
    stride_om, stride_on,      # output strides
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused SwiGLU: output = silu(input[:, :N]) * input[:, N:2*N].

    Processes [BLOCK_M, BLOCK_N] tiles. Input shape [M, 2*N], output [M, N].
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Load gate (first half) and up (second half)
    gate_ptrs = input_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
    up_ptrs = input_ptr + offs_m[:, None] * stride_im + (offs_n[None, :] + N) * stride_in

    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)

    # SiLU(gate) * up
    silu_gate = gate * tl.sigmoid(gate)
    result = silu_gate * up

    # Store
    out_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, result, mask=mask)
