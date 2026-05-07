"""Matrix multiplication — pure @triton.jit kernel with autotune.

Ported from Triton official tutorial (BSD license)
https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
adapted for NeuroBrix:
  - NBXTensor inputs (handled by wrappers, kernel sees raw ptrs/strides)
  - Volta-aware static dtype path (PROMOTE_B + IEEE_PRECISION constexpr)
  - @triton.autotune across 18 configs (Phase 1.5, 2026-05): the only
    proven path to ≥70% cuBLAS HMMA on Sana DiT shapes — see CLAUDE.md
    "Autotune policy" section for the doctrinal exception that allows
    @triton.autotune on mm/bmm/addmm/conv2d.
  - tl.dot 3-arg HMMA-FMA fused form
  - tl.assume integer-analyzer hints
  - in-kernel cast accum → output dtype
Handles mm (2D) + addmm (with bias). bmm handled in wrappers via batch loop.
"""

import triton
import triton.language as tl

from ._autotune_policy import maybe_pin_single, is_matmul_pinned


# Per-architecture autotune configs (tutorial pattern
# `get_cuda_autotune_config()` adapted to NeuroBrix: gate by detected
# compute capability so each hardware explores ONLY its viable subspace).
# Volta sm_70 has 96 KB SMEM/SM (vs 192 KB sm_80 / 228 KB sm_90); large
# blocks (BM≥128 BK≥64 warps=8) saturate SMEM → register spill →
# catastrophic perf (98-145 ms measured Phase 1.5 Étape 2 FlagGems
# bench). The Volta-viable subspace is restricted to BM∈{32,64},
# BN∈{32,64,128}, BK∈{32,64}, warps∈{2,4}, stages∈{2..5} — ~20
# combinations giving the autotuner a denser space of fitting configs.
# Ampere+ (sm_80+) gates BM/BN/BK larger as the hardware supports it.

_MATMUL_AUTOTUNE_VOLTA = [
    # BM=32 row
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=2),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=2),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=2),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
    # BM=64 row (the historical NeuroBrix default neighborhood)
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=2),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
]

# Ampere/Hopper subspace (sm_80+) — larger blocks allowed thanks to
# SMEM doubling and tensor-core throughput. Tutorial canonical configs.
_MATMUL_AUTOTUNE_AMPERE_PLUS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
] + _MATMUL_AUTOTUNE_VOLTA  # Ampere+ also benefits from smaller configs


def _detect_arch_configs():
    """Return autotune configs for the active CUDA arch.

    sm_70 (Volta V100) → Volta-viable subset (no SMEM-saturating blocks).
    sm_80+ (Ampere/Hopper) → tutorial canonical + Volta subset.
    Cached at module import time. Falls back to Volta subset if detection
    fails (safe choice — won't OOM on any arch).
    """
    try:
        cap = triton.runtime.driver.active.get_current_target().arch
        # Triton arch on NVIDIA = compute cap × 10 (sm_70 → 70, sm_80 → 80)
        if isinstance(cap, int) and cap >= 80:
            return _MATMUL_AUTOTUNE_AMPERE_PLUS
        return _MATMUL_AUTOTUNE_VOLTA
    except Exception:
        return _MATMUL_AUTOTUNE_VOLTA


_MATMUL_AUTOTUNE_CONFIGS = maybe_pin_single(
    _detect_arch_configs(), is_matmul_pinned)


@triton.autotune(configs=_MATMUL_AUTOTUNE_CONFIGS,
                 key=['M', 'N', 'K', 'IEEE_PRECISION', 'PROMOTE_B'],
                 cache_results=True)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    IEEE_PRECISION: tl.constexpr = False,
    PROMOTE_B: tl.constexpr = False,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 32,
    GROUP_M: tl.constexpr = 8,
):
    """C = A @ B where A is [M, K], B is [K, N], C is [M, N].

    Accumulates in fp32 for numerical stability.
    Output dtype determined by output pointer's dtype (in-kernel cast).

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

    Phase 1.5 (2026-05): @triton.autotune ENABLED. The autotune key
    includes IEEE_PRECISION + PROMOTE_B so each (Volta-fp32 / Volta-fp16-mixed
    / Ampere+ pure fp16) path gets its own selected config.
    """
    # tl.assume hints (tutorial pattern) — combined with autotune they
    # carry through; without autotune they were inactive (rollback 396fef1).
    tl.assume(stride_am > 0); tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0); tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0); tl.assume(stride_cn > 0)
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
        # 3-arg HMMA-FMA fused form (tutorial pattern).
        if IEEE_PRECISION:
            accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
        else:
            accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # In-kernel cast accum → output dtype (tutorial pattern; faster than
    # tl.store auto-cast since the conversion happens in registers).
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(configs=_MATMUL_AUTOTUNE_CONFIGS,
                 key=['M', 'N', 'K', 'IEEE_PRECISION', 'PROMOTE_B'],
                 cache_results=True)
@triton.jit
def addmm_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    alpha, beta,
    IEEE_PRECISION: tl.constexpr = False,
    PROMOTE_B: tl.constexpr = False,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 32,
    GROUP_M: tl.constexpr = 8,
):
    """C = beta * bias + alpha * (A @ B) where bias is [N].

    PROMOTE_B: see matmul_kernel docstring. Same in-kernel fp16→fp32
    tile cast; enables the wrapper to keep fp16 weights fp16 in memory
    while still running tl.dot with matched dtypes.

    Phase 1.5 (2026-05): @triton.autotune ENABLED — same configs as
    matmul_kernel. Tutorial pattern adapted with NeuroBrix's bias
    addition + alpha/beta scaling.
    """
    tl.assume(stride_am > 0); tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0); tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0); tl.assume(stride_cn > 0)
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
            accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
        else:
            accumulator = tl.dot(a, b, accumulator)
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
