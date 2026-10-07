"""GEMM on the m8n8k4 matrix unit — `mm` and `addmm` where the hardware profile declares `matrix_unit`.

Same contract as `matmul_kernel` / `addmm_kernel` (ops/matmul.py) for operands that are both the unit's operand
dtype in memory: C = A @ B (+ beta * bias + alpha scaling for addmm), fp32 accumulation, any strides on B (a
pre-transposed weight is read in place), int64 offsets, M/N/K tails masked, the output cast to C's dtype, and the
fused epilogues (silu, gelu exact, gelu tanh) with the per-stage rounding emulation of the unfused pair: the
accumulator is rounded to C's dtype, widened back, then the standalone kernel's formula is applied.

The products of two fp16 operands are exact in fp32, so on such operands this kernel computes the same sums as
the fp32 `tl.dot(input_precision="ieee")` path, in another accumulation order. The tile is the profile's
(`matrix_unit.mm`), never tuned at run time: no autotune key, nothing to certify.

Operands go global -> registers (coalesced) -> shared memory as [k-step x rows, 4] -> HMMA fragments
(ops/mma_m8n8k4.py); the next K tile is loaded while the current one is multiplied.
"""

import functools

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from .mma_m8n8k4 import (acc_layout, acc_rows_cols, frag_a_layout, frag_b_layout, mma_k4, stage_k_inner,
                         stage_k_outer)


@gluon.jit
def _sigmoid(x):
    return 1.0 / (1.0 + gl.exp(-x))


@gluon.jit
def _tanh(x):
    return 2.0 * _sigmoid(2.0 * x) - 1.0


@gluon.jit
def matmul_m8n8k4_kernel(
    A, B, Bias, C,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    alpha, beta,
    HAS_BIAS: gl.constexpr, PROMOTE_BIAS: gl.constexpr, EPILOGUE: gl.constexpr, B_K_CONTIG: gl.constexpr,
    MB: gl.constexpr, NB: gl.constexpr, BK: gl.constexpr, GROUP_M: gl.constexpr,
    L: gl.constexpr, LA: gl.constexpr, LB: gl.constexpr, GA: gl.constexpr, GB: gl.constexpr,
):
    BM: gl.constexpr = 16 * MB
    BN: gl.constexpr = 16 * NB
    J: gl.constexpr = BK // 4
    SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

    pid = gl.program_id(0).to(gl.int64)
    num_pid_m = gl.cdiv(M, BM)
    num_pid_n = gl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    first_pid_m = (pid // num_pid_in_group) * GROUP_M
    group_size_m = gl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    m0 = pid_m * BM
    n0 = pid_n * BN

    sa = gl.allocate_shared_memory(gl.float16, [J * BM, 4], SH)
    sb = gl.allocate_shared_memory(gl.float16, [J * BN, 4], SH)

    # A [BM, BK], k along the columns (the wrapper makes A row-major).
    ra = gl.arange(0, BM, layout=gl.SliceLayout(1, GA))[:, None]
    ka = gl.arange(0, BK, layout=gl.SliceLayout(0, GA))[None, :]
    pa = A + (m0 + ra) * stride_am + ka * stride_ak
    ma = (m0 + ra) < M
    # B either [BN, BK] (k contiguous: a pre-transposed weight) or [BK, BN] (n contiguous).
    if B_K_CONTIG:
        nb_ = gl.arange(0, BN, layout=gl.SliceLayout(1, GB))[:, None]
        kb = gl.arange(0, BK, layout=gl.SliceLayout(0, GB))[None, :]
    else:
        kb = gl.arange(0, BK, layout=gl.SliceLayout(1, GB))[:, None]
        nb_ = gl.arange(0, BN, layout=gl.SliceLayout(0, GB))[None, :]
    pb = B + kb * stride_bk + (n0 + nb_) * stride_bn
    mb_ = (n0 + nb_) < N

    acc = gl.zeros([MB, 2, 8, NB, 2, 2, 4], gl.float32, layout=L)
    ta = gl.load(pa, mask=ma & (ka < K), other=0.0)
    tb = gl.load(pb, mask=mb_ & (kb < K), other=0.0)
    for k0 in range(0, K, BK):
        gl.barrier()                                       # every warp is done reading the previous tile
        stage_k_inner(sa, ta, BM, BK)
        if B_K_CONTIG:
            stage_k_inner(sb, tb, BN, BK)
        else:
            stage_k_outer(sb, tb, BK, BN)
        gl.barrier()
        if k0 + BK < K:                                    # the next tile loads while the HMMAs run
            ta = gl.load(pa + (k0 + BK) * stride_ak, mask=ma & (ka + k0 + BK < K), other=0.0)
            tb = gl.load(pb + (k0 + BK) * stride_bk, mask=mb_ & (kb + k0 + BK < K), other=0.0)
        for j in gl.static_range(J):
            acc = mma_k4(sa, j * BM, sb, j * BN, acc, MB, NB, LA, LB, L)

    rows, cols = acc_rows_cols(L, MB, NB)
    rows = m0 + rows
    cols = n0 + cols
    if HAS_BIAS:
        bias = gl.load(Bias + cols, mask=cols < N, other=0.0)
        if PROMOTE_BIAS:
            bias = bias.to(gl.float32)
        acc = alpha * acc + beta * bias
    c = acc.to(C.dtype.element_ty)
    if EPILOGUE != 0:
        # Per-stage rounding emulation, as matmul_kernel: `c` carries the unfused GEMM's store rounding, the
        # widening is the standalone epilogue kernel's load; the formulas are those of ops/silu.py / ops/gelu.py.
        x = c.to(gl.float32)
        if EPILOGUE == 1:
            out = x * _sigmoid(x)
        elif EPILOGUE == 2:
            out = 0.5 * (1.0 + gl.erf(0.707106781 * x)) * x
        else:
            out = 0.5 * (1.0 + _tanh(0.7978845608 * x * (1.0 + 0.044715 * x * x))) * x
        c = out.to(C.dtype.element_ty)
    gl.store(C + rows.to(gl.int64) * stride_cm + cols.to(gl.int64) * stride_cn, c, mask=(rows < M) & (cols < N))


@functools.lru_cache(maxsize=None)
def mm_layouts(MB: int, NB: int, WM: int, WN: int, BK: int, b_k_contig: bool) -> dict:
    """Every layout the kernel takes for a tile of 16 MB x 16 NB x BK over WM x WN warps."""
    nw = WM * WN
    tk = min(BK // 8, 32)
    g_rows = gl.BlockedLayout([1, 8], [32 // tk, tk], [nw, 1], [1, 0])        # [rows, k], 8 halves along k
    return dict(
        L=acc_layout(MB, NB, WM, WN), LA=frag_a_layout(MB, NB, WM, WN), LB=frag_b_layout(MB, NB, WM, WN),
        GA=g_rows,
        # [n, k] like A, or [k, n] with 4 consecutive k per thread (one STS.64 per transposing store)
        GB=g_rows if b_k_contig else gl.BlockedLayout([4, 2], [1, 32], [nw, 1], [1, 0]))
