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

from ._configs import (arch_smem_budget, configs_within_smem_budget,
                       nbx_autotune)
import triton.language as tl

from ._autotune_policy import maybe_pin_single, is_matmul_pinned
from ._common import sigmoid, tanh_fn

# Vertical-fusion epilogue codes (fusion_vertical pass, EXACT gate).
# 0 = none, 1 = silu, 2 = gelu exact, 3 = gelu tanh. The epilogue is a
# tl.constexpr NOT in the autotune key: both fused and unfused variants
# share ONE tuning-cache entry per (M, N, K, IEEE, PROMOTE_B), so the
# selected config — and therefore the tl.dot accumulation order — is
# identical by construction (the two-leg byte-exactness contract: the
# R16 research showed rounding emulation alone is insufficient if the
# fused and unfused kernels can tune to different BLOCK_K).
#
# Byte-exactness emulation: the unfused pair is
#   matmul_kernel: fp32 acc -> cast to C dtype -> store        (round 1)
#   silu/gelu kernel: load C dtype -> .to(fp32) -> f(x) -> store (round 2)
# The fused epilogue reproduces BOTH rounding points in-register:
#   acc -> .to(C dtype) -> .to(fp32) -> f(x) -> store
# using the SAME formula text and the SAME _common helpers as the
# standalone kernels, so the arithmetic lowers identically.
EPILOGUE_NONE: int = 0
EPILOGUE_SILU: int = 1
EPILOGUE_GELU_EXACT: int = 2
EPILOGUE_GELU_TANH: int = 3


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
    """Return the autotune config space for the active architecture.

    WHY THIS ASKS THE DRIVER, when R23/R24 put hardware parameters in
    `config/vendors/<vendor>/<arch>.yml`: autotune configs are bound at
    DECORATION time, which is module import — before `set_hardware_profile()`
    has run at executor construction. There is no profile to read yet. This is
    the documented bootstrap exception, not a licence to query the driver
    elsewhere; every parameter that CAN wait for the profile reads the YAML
    (see `_safe_num_stages`, which moved to `pipelining.max_num_stages`).

    The spaces:

    * sm_80+ (Ampere/Hopper) — tutorial canonical plus the Volta subset.
    * sm_70 (Volta) — Volta-viable subset only: larger blocks saturate the
      96 KB SMEM, spill registers and collapse throughput (measured, Phase 1.5).
    * CDNA (gfx*) — the Volta-viable subset. AMD's Triton target reports `arch`
      as a STRING ("gfx90a"), so the old `isinstance(cap, int)` test silently
      dropped every AMD card into the Volta space *by accident*. It lands in the
      same place deliberately now, and for a stated reason: CDNA exposes 64 KB
      of LDS against Volta's 96 KB, so the small-block space is the safe one
      until an MI-series card measures a better one. Named as a first-light
      task, not left as a coincidence.
    * anything else (Apple, unknown) — the Volta subset as the STARTING space.

    Whichever space is chosen, it is then filtered against the executing
    hardware's `memory.max_shared_memory_per_block`, read from its vendor YAML.
    Choosing by architecture NAME alone was wrong in two directions and both
    are now decided rather than inherited:

    * **Apple** was sent to the Volta subset because that space "fits the
      smallest budget". It does not: Volta declares 96 KB and an Apple GPU has
      32 KB, so a BM=64/BN=128/BK=64 tile at 3 stages wants 72 KB and cannot
      run there. Nine of the seventeen Volta configs survive the real budget.
    * **CDNA** landed in the Volta space by accident (a string arch failed an
      `isinstance(cap, int)` test), was later made deliberate, and still had no
      budget applied — CDNA1/2 declare 64 KB, which excludes the largest Volta
      tile at 72 KB.

    Filtering never empties the space and never invents a budget: with no
    matching profile the space is returned untouched.
    """
    try:
        arch = triton.runtime.driver.active.get_current_target().arch
    except Exception:
        return _MATMUL_AUTOTUNE_VOLTA

    # NVIDIA: Triton reports compute capability x 10 (sm_70 -> 70, sm_80 -> 80)
    if isinstance(arch, int):
        space = (_MATMUL_AUTOTUNE_AMPERE_PLUS if arch >= 80
                 else _MATMUL_AUTOTUNE_VOLTA)
    else:
        # AMD CDNA/RDNA and any other string-arch target (Apple included).
        space = _MATMUL_AUTOTUNE_VOLTA

    return configs_within_smem_budget(space, arch_smem_budget())


_MATMUL_AUTOTUNE_CONFIGS = maybe_pin_single(
    _detect_arch_configs(), is_matmul_pinned)


@nbx_autotune(configs=_MATMUL_AUTOTUNE_CONFIGS,
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
    EPILOGUE: tl.constexpr = 0,
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
    if EPILOGUE != 0:
        # Per-stage rounding emulation: `c` already carries round 1 (the
        # unfused mm's store cast); .to(fp32) is the standalone epilogue
        # kernel's lossless load upcast. Formulas are copied VERBATIM
        # from ops/silu.py / ops/gelu.py; the store's implicit downcast
        # mirrors the standalone kernel's tl.store.
        x_fp32 = c.to(tl.float32)
        if EPILOGUE == 1:
            out = x_fp32 * sigmoid(x_fp32)
        elif EPILOGUE == 2:
            cdf = 0.5 * (1.0 + tl.math.erf(0.707106781 * x_fp32))
            out = cdf * x_fp32
        else:
            cdf = 0.5 * (1.0 + tanh_fn(0.7978845608 * x_fp32 * (1.0 + 0.044715 * x_fp32 * x_fp32)))
            out = cdf * x_fp32
        tl.store(c_ptrs, out, mask=c_mask)
    else:
        tl.store(c_ptrs, c, mask=c_mask)


@nbx_autotune(configs=_MATMUL_AUTOTUNE_CONFIGS,
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
    EPILOGUE: tl.constexpr = 0,
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
    if EPILOGUE != 0:
        # Per-stage rounding emulation (see matmul_kernel): the unfused
        # addmm stores the fp32 accumulator with tl.store's implicit
        # downcast (round 1); the standalone epilogue kernel loads and
        # upcasts to fp32. Reproduce both in-register, then apply the
        # VERBATIM standalone formulas.
        x_fp32 = accumulator.to(c_ptr.dtype.element_ty).to(tl.float32)
        if EPILOGUE == 1:
            out = x_fp32 * sigmoid(x_fp32)
        elif EPILOGUE == 2:
            cdf = 0.5 * (1.0 + tl.math.erf(0.707106781 * x_fp32))
            out = cdf * x_fp32
        else:
            cdf = 0.5 * (1.0 + tanh_fn(0.7978845608 * x_fp32 * (1.0 + 0.044715 * x_fp32 * x_fp32)))
            out = cdf * x_fp32
        tl.store(c_ptrs, out, mask=c_mask)
    else:
        tl.store(c_ptrs, accumulator, mask=c_mask)
