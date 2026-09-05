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
    # A block beyond the padded token count reads and writes nothing: its
    # sorted-token and expert entries are not written by moe_align, so every
    # access below carries `block_valid` (no early exit — unstructured
    # control flow has no lowering on every backend).
    block_valid = pid_m * BLOCK_SIZE_M < num_tokens_post_padded

    # Load sorted token IDs for this block
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id, mask=block_valid, other=0).to(tl.int64)
    token_mask = (offs_token < num_valid_tokens) & block_valid

    # Load expert id → load absolute weight pointer from table
    off_experts = tl.load(expert_ids_ptr + pid_m, mask=block_valid, other=0)
    expert_ptr_int = tl.load(expert_ptrs_ptr + off_experts, mask=block_valid, other=0)
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
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & block_valid,
            other=0.0,
        )
        # NOTE: Phase 1.5 Étape 1 (2026-05) tested 3-arg `tl.dot(a, b, acc)`
        # form — measured 0% gain on V100 sm_70. Reverted to 2-arg form.
        # See kernels/ops/matmul.py:matmul_kernel docstring for full audit.
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
def fused_moe_wna16_kernel(
    a_ptr,                          # activations [M, K]
    qw_ptrs_ptr,                    # [E] int64 — packed int4 [K//8, N] per expert
    sc_ptrs_ptr,                    # [E] int64 — scales fp16 [K//G, N]
    mn_ptrs_ptr,                    # [E] int64 — mins fp16 [K//G, N]
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak,
    stride_qk, stride_qn,           # packed strides (rows of int32)
    stride_sg, stride_sn,           # scales/mins strides
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    QGROUP: tl.constexpr,           # quant group size along K (128)
    QPACK: tl.constexpr,            # nibbles per int32 (8)
    TOPK_DIVIDE: tl.constexpr = True,
):
    """Grouped GEMM over int4-g128-asym experts — in-register dequant.

    Structure mirrors fused_moe_kernel exactly (same pid mapping, same
    K-loop, same routed-weight epilogue); the B tile is rebuilt each
    iteration from the expert's packed triplet with the tier's
    CANONICAL dequant (pure fp32: q*scale+min — contraction-immune,
    int4 x fp16 products exact in fp32) and the dot runs fp32 x fp32
    (on sm_70 fp16 tl.dot lowers to FMA fp32 anyway — Phase 1.5
    audit). BLOCK_SIZE_K must divide QGROUP so a K-tile never crosses
    a quant-group boundary (one scales/mins row vector per
    iteration). SPLIT_K = 1, no atomics (tier byte contract). The
    byte oracle is fused_moe_fp32b_kernel below — textually identical
    minus the unpack.
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
    # A block beyond the padded token count reads and writes nothing: its
    # sorted-token and expert entries are not written by moe_align, so every
    # access below carries `block_valid` (no early exit — unstructured
    # control flow has no lowering on every backend).
    block_valid = pid_m * BLOCK_SIZE_M < num_tokens_post_padded

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id, mask=block_valid, other=0).to(tl.int64)
    token_mask = (offs_token < num_valid_tokens) & block_valid

    off_experts = tl.load(expert_ids_ptr + pid_m, mask=block_valid, other=0)
    qw_base = tl.cast(tl.load(qw_ptrs_ptr + off_experts, mask=block_valid, other=0),
                      tl.pointer_type(tl.int32), bitcast=True)
    sc_base = tl.cast(tl.load(sc_ptrs_ptr + off_experts, mask=block_valid, other=0),
                      tl.pointer_type(tl.float16), bitcast=True)
    mn_base = tl.cast(tl.load(mn_ptrs_ptr + off_experts, mask=block_valid, other=0),
                      tl.pointer_type(tl.float16), bitcast=True)

    offs_bn = (pid_n * BLOCK_SIZE_N
               + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

    if TOPK_DIVIDE:
        a_idx = offs_token[:, None] // top_k
    else:
        a_idx = offs_token[:, None]
    a_ptrs = a_ptr + (a_idx * stride_am + offs_k[None, :] * stride_ak)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_rem = K - k * BLOCK_SIZE_K
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < k_rem),
            other=0.0,
        )
        cur_k = k * BLOCK_SIZE_K + offs_k
        packed = tl.load(
            qw_base + (cur_k[:, None] // QPACK) * stride_qk
            + offs_bn[None, :] * stride_qn,
            mask=(offs_k[:, None] < k_rem) & block_valid, other=0)
        q = (packed >> ((cur_k[:, None] % QPACK) * 4)) & 0xF
        g = k * BLOCK_SIZE_K // QGROUP  # BLOCK_SIZE_K divides QGROUP
        scale = tl.load(sc_base + g * stride_sg + offs_bn * stride_sn, mask=block_valid, other=0.0)
        mn = tl.load(mn_base + g * stride_sg + offs_bn * stride_sn, mask=block_valid, other=0.0)
        # Canonical dequant — the tier's contraction-immune fp32 form,
        # textually identical to the GEMV family's.
        b = (q.to(tl.float32) * scale[None, :].to(tl.float32)
             + mn[None, :].to(tl.float32))
        b = tl.where(offs_k[:, None] < k_rem, b, 0.0)
        accumulator += tl.dot(a.to(tl.float32), b)
        a_ptrs += BLOCK_SIZE_K * stride_ak

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
def fused_moe_fp32b_kernel(
    a_ptr,
    expert_ptrs_ptr,                # [E] int64 — dense FP32 [K, N] per expert
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    TOPK_DIVIDE: tl.constexpr = True,
):
    """The wna16 kernel's BYTE ORACLE: dense fp32 expert weights (the
    family's standalone dequant output), identical pid mapping, K-loop,
    fp32 dot and epilogue — textually the kernel above minus the
    unpack. Never used on a hot path; exists so the fused W4 kernel is
    provably byte-identical to dequantize-then-grouped-GEMM."""
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
    # A block beyond the padded token count reads and writes nothing: its
    # sorted-token and expert entries are not written by moe_align, so every
    # access below carries `block_valid` (no early exit — unstructured
    # control flow has no lowering on every backend).
    block_valid = pid_m * BLOCK_SIZE_M < num_tokens_post_padded

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id, mask=block_valid, other=0).to(tl.int64)
    token_mask = (offs_token < num_valid_tokens) & block_valid

    off_experts = tl.load(expert_ids_ptr + pid_m, mask=block_valid, other=0)
    b_base = tl.cast(tl.load(expert_ptrs_ptr + off_experts, mask=block_valid, other=0),
                     tl.pointer_type(tl.float32), bitcast=True)

    offs_bn = (pid_n * BLOCK_SIZE_N
               + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

    if TOPK_DIVIDE:
        a_idx = offs_token[:, None] // top_k
    else:
        a_idx = offs_token[:, None]
    a_ptrs = a_ptr + (a_idx * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_base + (offs_k[:, None] * stride_bk
                       + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_rem = K - k * BLOCK_SIZE_K
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < k_rem),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < k_rem) & block_valid,
            other=0.0,
        )
        accumulator += tl.dot(a.to(tl.float32), b)
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
