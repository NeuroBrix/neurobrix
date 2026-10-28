"""Convolution on the m8n8k4 matrix unit — `conv2d` where the hardware profile declares `matrix_unit`.

An implicit GEMM over the operands in place: per group g, C[m, n] = sum_k A[m, k] B[n, k] with
m = (batch, oh, ow), n = the group's output channel and k = (c, r, s) flattened in the weight's own order, so a
contiguous weight [OC, IC/g, KH, KW] is a k-contiguous B read in place (no copy, no im2col buffer) and A is gathered
from the input through its four strides (padding, stride and dilation are masks and index arithmetic). Same contract
as `conv2d_forward_kernel` (ops/conv2d.py) for operands that are both the unit's operand dtype in memory: fp32
accumulation (one K tile in the unit, which rounds toward zero, the tiles summed on the FMA units with round to
nearest — ops/matmul_m8n8k4.py), int64 offsets, every tail masked, the output cast to C's dtype; the bias, when given, is added to the
fp32 accumulator before that single rounding (the product of two fp16 values is exact in fp32, so the sums are the
tl.dot kernel's in another order). The tile is the profile's (`matrix_unit.mm`), never tuned at run time: no
autotune key, nothing to certify.

A is loaded with consecutive lanes on consecutive output rows (consecutive `ow`: coalesced at stride 1), each lane
holding 4 consecutive k — one 8-byte store into the [k-step x rows, 4] staging of ops/mma_m8n8k4.py.
"""

import functools

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from .mma_m8n8k4 import acc_rows_cols, mma_k4, stage_k_inner
from .matmul_m8n8k4 import mm_layouts


@gluon.jit
def conv2d_m8n8k4_kernel(
    X, Wt, Bias, C,
    M, NG, K, IH, IW, OH, OW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_cn, stride_cc, stride_ch, stride_cw,
    CG, OG,
    KH: gl.constexpr, KW: gl.constexpr, SH_: gl.constexpr, SW_: gl.constexpr, PH: gl.constexpr,
    PW: gl.constexpr, DH: gl.constexpr, DW: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    MB: gl.constexpr, NB: gl.constexpr, BK: gl.constexpr, GROUP_M: gl.constexpr,
    L: gl.constexpr, LA: gl.constexpr, LB: gl.constexpr, GA: gl.constexpr, GB: gl.constexpr,
):
    BM: gl.constexpr = 16 * MB
    BN: gl.constexpr = 16 * NB
    J: gl.constexpr = BK // 4
    KHW: gl.constexpr = KH * KW
    SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

    pid = gl.program_id(0).to(gl.int64)
    g = gl.program_id(1).to(gl.int64)                      # the group
    X = X + g * CG * stride_xc
    Wt = Wt + g * OG * stride_wo
    C = C + g * OG * stride_cc
    num_pid_m = gl.cdiv(M, BM)
    num_pid_n = gl.cdiv(NG, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    first_pid_m = (pid // num_pid_in_group) * GROUP_M
    group_size_m = gl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    m0 = pid_m * BM
    n0 = pid_n * BN

    sa = gl.allocate_shared_memory(gl.float16, [J * BM, 4], SH)
    sb = gl.allocate_shared_memory(gl.float16, [J * BN, 4], SH)

    # A rows: (batch, oh, ow) once per tile; the k columns are decoded per K tile against constexpr divisors.
    ma1 = m0 + gl.arange(0, BM, layout=gl.SliceLayout(1, GA))
    ow_ = ma1 % OW
    oh_ = (ma1 // OW) % OH
    xrow = ((ma1 // OW) // OH) * stride_xn
    ih0 = (oh_ * SH_ - PH)[:, None]
    iw0 = (ow_ * SW_ - PW)[:, None]
    ma = (ma1 < M)[:, None]
    xrow = xrow[:, None]
    ka1 = gl.arange(0, BK, layout=gl.SliceLayout(0, GA))
    # B [BN, BK], k contiguous: the weight read in place.
    nb_ = gl.arange(0, BN, layout=gl.SliceLayout(1, GB))[:, None]
    kb = gl.arange(0, BK, layout=gl.SliceLayout(0, GB))[None, :]
    pb = Wt + (n0 + nb_) * stride_wo + kb
    mb_ = (n0 + nb_) < NG

    acc = gl.zeros([MB, 2, 8, NB, 2, 2, 4], gl.float32, layout=L)
    ta = _gather(X, xrow, ih0, iw0, ma, ka1, 0, K, IH, IW, stride_xc, stride_xh, stride_xw, KHW, KW, DH, DW)
    tb = gl.load(pb, mask=mb_ & (kb < K), other=0.0)
    for k0 in range(0, K, BK):
        gl.barrier()                                       # every warp is done reading the previous tile
        stage_k_inner(sa, ta, BM, BK)
        stage_k_inner(sb, tb, BN, BK)
        gl.barrier()
        if k0 + BK < K:                                    # the next tile loads while the HMMAs run
            ta = _gather(X, xrow, ih0, iw0, ma, ka1, k0 + BK, K, IH, IW, stride_xc, stride_xh, stride_xw, KHW, KW,
                         DH, DW)
            tb = gl.load(pb + (k0 + BK), mask=mb_ & (kb + k0 + BK < K), other=0.0)
        t = gl.zeros([MB, 2, 8, NB, 2, 2, 4], gl.float32, layout=L)
        for j in gl.static_range(J):
            t = mma_k4(sa, j * BM, sb, j * BN, t, MB, NB, LA, LB, L)
        acc = acc + t                                      # across tiles on the FMA units, round to nearest

    rows, cols = acc_rows_cols(L, MB, NB)
    rows = m0 + rows
    cols = n0 + cols
    mask = (rows < M) & (cols < NG)
    if HAS_BIAS:
        bias = gl.load(Bias + g * OG + cols.to(gl.int64), mask=cols < NG, other=0.0)
        acc = acc + bias.to(gl.float32)
    r64 = rows.to(gl.int64)
    ow_o = r64 % OW
    oh_o = (r64 // OW) % OH
    n_o = (r64 // OW) // OH
    gl.store(C + n_o * stride_cn + oh_o * stride_ch + ow_o * stride_cw + cols.to(gl.int64) * stride_cc,
             acc.to(C.dtype.element_ty), mask=mask)


@gluon.jit
def _gather(X, xrow, ih0, iw0, ma, ka1, k0, K, IH, IW, stride_xc, stride_xh, stride_xw,
            KHW: gl.constexpr, KW: gl.constexpr, DH: gl.constexpr, DW: gl.constexpr):
    """A[rows, k0:k0+BK]: input[batch, c, oh*sh - ph + r*dh, ow*sw - pw + s*dw] for k = (c, r, s), 0 outside."""
    k = k0 + ka1
    c = k // KHW
    rs = k % KHW
    ih = ih0 + ((rs // KW) * DH)[None, :]
    iw = iw0 + ((rs % KW) * DW)[None, :]
    m = ma & (k < K)[None, :] & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
    p = X + xrow + (c.to(gl.int64) * stride_xc)[None, :] + ih.to(gl.int64) * stride_xh + iw.to(gl.int64) * stride_xw
    return gl.load(p, mask=m, other=0.0)


@functools.lru_cache(maxsize=None)
def conv_layouts(MB: int, NB: int, WM: int, WN: int, BK: int) -> dict:
    """The GEMM's layouts (ops/matmul_m8n8k4.py, B k-contiguous) with the gathered A walked row-fastest: 32
    consecutive output rows per warp, 4 consecutive k per lane."""
    lay = dict(mm_layouts(MB, NB, WM, WN, BK, True))
    lay["GA"] = gl.BlockedLayout([1, 4], [32, 1], [WM * WN, 1], [0, 1])
    return lay
