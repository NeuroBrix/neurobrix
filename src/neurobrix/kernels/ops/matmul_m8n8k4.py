"""GEMM on the m8n8k4 matrix unit — `mm` and `addmm` where the hardware profile declares `matrix_unit`.

Same contract as `matmul_kernel` / `addmm_kernel` / `baddbmm_kernel` (ops/matmul.py) for operands that are both the unit's operand
dtype in memory: C[z] = A[z] @ B[z] (+ beta * bias + alpha scaling; the bias broadcast by its strides; batch on
grid axis 1, stride 0 for mm / addmm), fp32 accumulation, any strides on B (a
pre-transposed weight is read in place), int64 offsets, M/N/K tails masked, the output cast to C's dtype, and the
fused epilogues (silu, gelu exact, gelu tanh) with the per-stage rounding emulation of the unfused pair: the
accumulator is rounded to C's dtype, widened back, then the standalone kernel's formula is applied.

The products of two fp16 operands are exact in fp32. The unit's own accumulation rounds toward zero (Fasi et al.
2021, PeerJ CS 7:e330), so it sums only one K tile into a zeroed accumulator; the tiles are summed on the FMA units
with round to nearest, as the fp32 `tl.dot(input_precision="ieee")` path sums, in another order. The tile is the profile's
(`matrix_unit.mm`), never tuned at run time: no autotune key, nothing to certify.

Operands go global -> registers (coalesced) -> shared memory as [k-step x rows, 4] -> HMMA fragments
(ops/mma_m8n8k4.py); the next K tile is loaded while the current one is multiplied.

SPLIT representation (SPLIT_A / SPLIT_B, decided by the DtypeEngine: triton/dtype.matrix_unit_representation, the
profile's `matrix_unit.fp32_split`): an operand of any float dtype in memory is widened to fp32, each row of its K tile (each
column for B) scaled by the power of two 2^s that brings that row's largest magnitude to [2^HI_EXP, 2^(HI_EXP+1)),
and carried as hi = fp16(x 2^s) and lo = fp16((x 2^s - hi) 2^LO_SHIFT). The scale is constant along k, so it factors
out of the dot product: a small row next to a large one keeps its own precision. Per K tile the unit computes hi*hi
into one zeroed accumulator and hi*lo + lo*hi into another; the tile's sum t_hh + t_lo 2^-LO_SHIFT, times 2^-s of
each split operand's row (column), is added to the fp32 accumulator on the FMA units (round to nearest). The unit's own accumulation rounds
toward zero (Fasi et al. 2021, PeerJ CS 7:e330), so it never carries the sum across tiles; lo*lo (< 2^-22 of the
product) is not formed (Ootomo & Yokota 2022, arXiv 2203.03341, eq. 24). Every power-of-two scaling is exact.
"""

import functools

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from .mma_m8n8k4 import (acc_layout, acc_rows_cols, bcast_rows, frag_a_layout, frag_b_layout, mma_k4,
                         stage_k_inner, stage_k_outer)


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
    stride_az, stride_am, stride_ak, stride_bz, stride_bk, stride_bn, stride_cz, stride_cm, stride_cn,
    stride_xz, stride_xm, stride_xn,
    alpha, beta,
    HAS_BIAS: gl.constexpr, PROMOTE_BIAS: gl.constexpr, EPILOGUE: gl.constexpr, B_K_CONTIG: gl.constexpr,
    SPLIT_A: gl.constexpr, SPLIT_B: gl.constexpr, HI_EXP: gl.constexpr, LO_SHIFT: gl.constexpr,
    MB: gl.constexpr, NB: gl.constexpr, BK: gl.constexpr, GROUP_M: gl.constexpr,
    L: gl.constexpr, LA: gl.constexpr, LB: gl.constexpr, GA: gl.constexpr, GB: gl.constexpr,
):
    BM: gl.constexpr = 16 * MB
    BN: gl.constexpr = 16 * NB
    J: gl.constexpr = BK // 4
    SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

    pid = gl.program_id(0).to(gl.int64)
    z = gl.program_id(1).to(gl.int64)                      # the batch (0 for mm / addmm)
    A = A + z * stride_az
    B = B + z * stride_bz
    C = C + z * stride_cz
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
    if SPLIT_A:
        sal = gl.allocate_shared_memory(gl.float16, [J * BM, 4], SH)
    if SPLIT_B:
        sbl = gl.allocate_shared_memory(gl.float16, [J * BN, 4], SH)

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
        if SPLIT_A:
            inv_a = _stage_split(sa, sal, ta, True, BM, BK, HI_EXP, LO_SHIFT)
        else:
            stage_k_inner(sa, ta, BM, BK)
        if SPLIT_B:
            inv_b = _stage_split(sb, sbl, tb, B_K_CONTIG, BN, BK, HI_EXP, LO_SHIFT)
        elif B_K_CONTIG:
            stage_k_inner(sb, tb, BN, BK)
        else:
            stage_k_outer(sb, tb, BK, BN)
        gl.barrier()
        if k0 + BK < K:                                    # the next tile loads while the HMMAs run
            ta = gl.load(pa + (k0 + BK) * stride_ak, mask=ma & (ka + k0 + BK < K), other=0.0)
            tb = gl.load(pb + (k0 + BK) * stride_bk, mask=mb_ & (kb + k0 + BK < K), other=0.0)
        if SPLIT_A or SPLIT_B:
            t_hh = gl.zeros([MB, 2, 8, NB, 2, 2, 4], gl.float32, layout=L)
            t_lo = gl.zeros([MB, 2, 8, NB, 2, 2, 4], gl.float32, layout=L)
            for j in gl.static_range(J):
                t_hh = mma_k4(sa, j * BM, sb, j * BN, t_hh, MB, NB, LA, LB, L)
                if SPLIT_A:
                    t_lo = mma_k4(sal, j * BM, sb, j * BN, t_lo, MB, NB, LA, LB, L)
                if SPLIT_B:
                    t_lo = mma_k4(sa, j * BM, sbl, j * BN, t_lo, MB, NB, LA, LB, L)
            t = t_hh + t_lo * (1.0 / (1 << LO_SHIFT))
            if SPLIT_A:
                t = t * _bcast_rows(gl.reshape(inv_a, [MB, 2, 8]), L)
            if SPLIT_B:
                t = t * _bcast_cols(gl.reshape(inv_b, [NB, 2, 2, 4]), L)
            acc = acc + t
        else:
            t = gl.zeros([MB, 2, 8, NB, 2, 2, 4], gl.float32, layout=L)
            for j in gl.static_range(J):
                t = mma_k4(sa, j * BM, sb, j * BN, t, MB, NB, LA, LB, L)
            acc = acc + t                                  # across tiles on the FMA units, round to nearest

    rows, cols = acc_rows_cols(L, MB, NB)
    rows = m0 + rows
    cols = n0 + cols
    if HAS_BIAS:
        # any broadcast of the bias: a [N] row has (0, 0, 1) strides, a full [B, M, N] its own
        bias = gl.load(Bias + z * stride_xz + rows.to(gl.int64) * stride_xm + cols.to(gl.int64) * stride_xn,
                       mask=(rows < M) & (cols < N), other=0.0)
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


@gluon.jit
def _pow2(e):
    """2^e as fp32 for an integer exponent in [-126, 127], built from its bits (exact)."""
    return ((e + 127) << 23).to(gl.float32, bitcast=True)


@gluon.jit
def _stage_split(s_hi, s_lo, tile, K_INNER: gl.constexpr, R: gl.constexpr, KK: gl.constexpr,
                 HI_EXP: gl.constexpr, LO_SHIFT: gl.constexpr):
    """Stage one operand tile as hi + lo of the unit's operand dtype, each row (k along the columns) or column (k
    along the rows) under its own power-of-two scale 2^s; returns 2^-s per row / column [R]. A zero row keeps
    s = 0; s is clamped to the fp32 normal range. A value hi already holds whole (an infinity among them) has lo 0."""
    x = tile.to(gl.float32)
    KA: gl.constexpr = 1 if K_INNER else 0
    m = gl.max(gl.abs(x), axis=KA)
    e = ((m.to(gl.int32, bitcast=True) >> 23) & 0xFF) - 127
    s = gl.where(m == 0.0, 0, HI_EXP - e)
    s = gl.minimum(gl.maximum(s, -126), 126)
    x = x * gl.expand_dims(_pow2(s), KA)
    hi = x.to(gl.float16)
    h32 = hi.to(gl.float32)
    lo = gl.where(h32 == x, 0.0, (x - h32) * (1 << LO_SHIFT)).to(gl.float16)
    if K_INNER:
        stage_k_inner(s_hi, hi, R, KK)
        stage_k_inner(s_lo, lo, R, KK)
    else:
        stage_k_outer(s_hi, hi, KK, R)
        stage_k_outer(s_lo, lo, KK, R)
    return _pow2(-s)


@gluon.jit
def _bcast_rows(r, L: gl.constexpr):
    """A [MB, 2, 8] row statistic in any layout -> broadcastable over L."""
    RL: gl.constexpr = gl.SliceLayout(3, gl.SliceLayout(4, gl.SliceLayout(5, gl.SliceLayout(6, L))))
    return bcast_rows(gl.convert_layout(r, RL), L)


@gluon.jit
def _bcast_cols(c, L: gl.constexpr):
    """A [NB, 2, 2, 4] column statistic (column = 16 o3 + 8 o4 + 4 o5 + o6, acc_rows_cols) -> broadcastable over L."""
    CL: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, L)))
    c = gl.convert_layout(c, CL)
    return gl.expand_dims(gl.expand_dims(gl.expand_dims(c, 0), 1), 2)


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
