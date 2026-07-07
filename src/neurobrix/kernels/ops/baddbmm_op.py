"""Batched matmul with optional bias (baddbmm/bmm) — pure @triton.jit kernel.

Originally ported from FlagGems baddbmm (Apache-2.0 license); body aligned
on kernels/ops/matmul.py:matmul_kernel (P-KERNEL-LAUNCH-HYGIENE B1,
2026-07) so the SAME batched kernel serves both:

  - aten::baddbmm  (HAS_BIAS=True):  out = beta * bias + alpha * (b1 @ b2)
  - aten::bmm      (HAS_BIAS=False): out = A @ B, one 3D-grid launch for the
    whole batch instead of the former per-batch (and per-row in decode)
    Python launch loops in wrappers.bmm.

batch1 is [B, M, K], batch2 is [B, K, N], bias broadcastable to [B, M, N].
Grid: (cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N), 1, B). Accumulates in fp32.

IEEE_PRECISION / PROMOTE_B carry the exact matmul_kernel semantics (see its
docstring): ieee dot for fp32-promoted inputs on pre-Ampere, in-register
fp16→fp32 b-tile promotion for the mixed activation/weight case. The K-loop
body (3-arg fused `tl.dot`, %-wrap row/col offsets, K-tail masks) is a
line-for-line mirror of matmul_kernel so a batched launch is value-identical
to the former per-batch matmul_kernel loop under the same autotune config.

Phase 1.5 (2026-05): @triton.autotune ENABLED with the same Volta-viable
configs as kernels/ops/matmul.py. See CLAUDE.md "Autotune policy" section
for the doctrinal exception.
"""

import triton
import triton.language as tl

from neurobrix.kernels.ops.matmul import _MATMUL_AUTOTUNE_CONFIGS


@triton.autotune(configs=_MATMUL_AUTOTUNE_CONFIGS,
                 key=['M', 'N', 'K', 'IEEE_PRECISION', 'PROMOTE_B',
                      'HAS_BIAS'],
                 cache_results=True)
@triton.jit
def baddbmm_kernel(
    A_ptr,
    B_ptr,
    out_ptr,
    bias_ptr,
    alpha,
    beta,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_ob, stride_om, stride_on,
    bias_batch_stride,
    bias_m_stride,
    bias_n_stride,
    IEEE_PRECISION: tl.constexpr = False,
    PROMOTE_B: tl.constexpr = False,
    HAS_BIAS: tl.constexpr = True,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 32,
    GROUP_M: tl.constexpr = 8,
):
    """out[b] = beta*bias[b] + alpha*(A[b] @ B[b])   (HAS_BIAS=True)
       out[b] = A[b] @ B[b]                          (HAS_BIAS=False)

    Grid: (cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N), 1, batch)
    Accumulates in fp32 for numerical stability; in-register cast to the
    output pointer's dtype before store (matmul_kernel pattern).
    """
    # Batch offset
    pid_b = tl.program_id(2)
    A_ptr += pid_b * stride_ab
    B_ptr += pid_b * stride_bb
    out_ptr += pid_b * stride_ob
    if HAS_BIAS:
        bias_ptr += pid_b * bias_batch_stride

    # tl.assume integer-analyzer hints (matmul_kernel pattern). Batch and
    # bias strides are excluded: they are legitimately 0 for broadcast.
    tl.assume(stride_am > 0); tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0); tl.assume(stride_bn > 0)
    tl.assume(stride_om > 0); tl.assume(stride_on > 0)

    # 2D tile indexing with grouping for L2 locality
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
    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Accumulate matmul in fp32 — line-for-line matmul_kernel K loop.
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        if PROMOTE_B:
            b = b.to(a.dtype)
        # 3-arg HMMA-FMA fused form (tutorial pattern) — same as
        # matmul_kernel so the bmm path stays bit-identical to the former
        # per-batch loop. IEEE_PRECISION supersedes the historical
        # `allow_tf32=False` (equivalent lowering for fp32 inputs).
        if IEEE_PRECISION:
            accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
        else:
            accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if HAS_BIAS:
        # Load bias and apply alpha/beta
        bias_ptrs = (bias_ptr + offs_cm[:, None] * bias_m_stride
                     + offs_cn[None, :] * bias_n_stride)
        bi = tl.load(bias_ptrs, mask=out_mask, other=0.0)
        accumulator = accumulator * alpha + bi * beta

    # In-kernel cast accum → output dtype (matmul_kernel pattern).
    c = accumulator.to(out_ptr.dtype.element_ty)
    o_ptrs = out_ptr + offs_cm[:, None] * stride_om + offs_cn[None, :] * stride_on
    tl.store(o_ptrs, c, mask=out_mask)
