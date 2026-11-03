"""The fused mixture-of-experts grouped GEMM on the m8n8k4 matrix unit — `fused_moe_kernel`'s contract where the
hardware profile declares `matrix_unit` and the experts are the unit's operand dtype in memory.

Same contract as `fused_moe_kernel` (ops/fused_moe.py): the routing alignment's blocks of sorted (token, expert)
pairs, one expert per block whose weights are read through the absolute pointer table, the activation row of each
pair (the token, or the pair itself without the top-k division), the routed weight on the down projection, the output
row written at the pair. The block of rows is the alignment's and the unit tile's at once: the rows of the
`matrix_unit.mm` row the certificate chose (`moe_wrappers.invoke_fused_moe`, the routing aligned at that block).

The products of two fp16 operands are exact in fp32; the unit sums one K tile into a zeroed accumulator and the tiles
are summed on the FMA units with round to nearest, as `matmul_m8n8k4_kernel` does (the same staging, fragments and
HMMA, ops/mma_m8n8k4.py).
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from .mma_m8n8k4 import acc_rows_cols, mma_k4, stage_k_inner, stage_k_outer


@gluon.jit
def fused_moe_m8n8k4_kernel(
    A, expert_ptrs, C, topk_w, sorted_ids, expert_ids, num_post_padded,
    N, K, EM, num_valid,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    MUL_ROUTED_WEIGHT: gl.constexpr, TOP_K: gl.constexpr, TOPK_DIVIDE: gl.constexpr, B_K_CONTIG: gl.constexpr,
    MB: gl.constexpr, NB: gl.constexpr, BK: gl.constexpr, GROUP_M: gl.constexpr,
    L: gl.constexpr, LA: gl.constexpr, LB: gl.constexpr, GA: gl.constexpr, GB: gl.constexpr,
):
    BM: gl.constexpr = 16 * MB
    BN: gl.constexpr = 16 * NB
    J: gl.constexpr = BK // 4
    SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

    pid = gl.program_id(0).to(gl.int64)
    num_pid_m = gl.cdiv(EM, BM)
    num_pid_n = gl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    first_pid_m = (pid // num_pid_in_group) * GROUP_M
    group_size_m = gl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    m0 = pid_m * BM
    n0 = pid_n * BN
    # a block past the padded pair count reads and writes nothing (its alignment entries are not written)
    block_valid = m0 < gl.load(num_post_padded)

    sa = gl.allocate_shared_memory(gl.float16, [J * BM, 4], SH)
    sb = gl.allocate_shared_memory(gl.float16, [J * BN, 4], SH)

    # A: the block's pairs' activation rows [BM, BK], k along the columns
    ra = gl.arange(0, BM, layout=gl.SliceLayout(1, GA))
    pair_a = gl.load(sorted_ids + m0 + ra, mask=(ra >= 0) & block_valid, other=0).to(gl.int64)
    ok_a = (pair_a < num_valid) & block_valid
    if TOPK_DIVIDE:
        row_a = pair_a // TOP_K
    else:
        row_a = pair_a
    ka = gl.arange(0, BK, layout=gl.SliceLayout(0, GA))[None, :]
    pa = A + row_a[:, None] * stride_am + ka * stride_ak
    ma = ok_a[:, None]
    # B: the block's expert, through the pointer table
    expert = gl.load(expert_ids + pid_m, mask=block_valid, other=0)
    B = gl.cast(gl.load(expert_ptrs + expert, mask=block_valid, other=0), gl.pointer_type(gl.float16), bitcast=True)
    if B_K_CONTIG:
        nb_ = gl.arange(0, BN, layout=gl.SliceLayout(1, GB))[:, None]
        kb = gl.arange(0, BK, layout=gl.SliceLayout(0, GB))[None, :]
    else:
        kb = gl.arange(0, BK, layout=gl.SliceLayout(1, GB))[:, None]
        nb_ = gl.arange(0, BN, layout=gl.SliceLayout(0, GB))[None, :]
    pb = B + kb * stride_bk + (n0 + nb_) * stride_bn
    mb_ = ((n0 + nb_) < N) & block_valid

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
        t = gl.zeros([MB, 2, 8, NB, 2, 2, 4], gl.float32, layout=L)
        for j in gl.static_range(J):
            t = mma_k4(sa, j * BM, sb, j * BN, t, MB, NB, LA, LB, L)
        acc = acc + t                                      # across tiles on the FMA units, round to nearest

    rows, cols = acc_rows_cols(L, MB, NB)
    pair = gl.load(sorted_ids + m0 + rows, mask=(rows >= 0) & block_valid, other=0).to(gl.int64)
    ok = (pair < num_valid) & block_valid
    cols = n0 + cols
    if MUL_ROUTED_WEIGHT:
        acc = acc * gl.load(topk_w + pair, mask=ok, other=0.0).to(gl.float32)
    gl.store(C + pair * stride_cm + cols.to(gl.int64) * stride_cn, acc.to(C.dtype.element_ty), mask=ok & (cols < N))
