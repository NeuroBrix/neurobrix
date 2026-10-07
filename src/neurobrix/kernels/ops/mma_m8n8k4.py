"""The m8n8k4 matrix unit (sm_70 tensor cores, SASS HMMA.884) as a Gluon building block.

Triton >= 3.3 lowers `tl.dot` to scalar FMA below sm_80 (triton-lang/triton#5066), so an arch whose profile
declares `matrix_unit.shape: m8n8k4` gets its tensor cores from here: our own Gluon kernels issue
`mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32` through `inline_asm_elementwise`, with the operands placed in
registers by linear layouts. Method and proofs: docs/reference/tensor-cores-through-linear-layouts.md.

The fragment (CUTLASS cute/atom/mma_traits_sm70.hpp, SM70_8x8x4_F32F16F16F32_TN): lane = a0 + 2 a1 + 4 q0 + 8 q1
+ 16 b, thread t = a0 + 2 a1 + 4 b of quadpair q holds A row t and B column t (k = 0..3), and accumulator
register v = v0 + 2 v1 + 4 v2 holds D[m = a0 + 2 v1 + 4 b, n = v0 + 2 a1 + 4 v2]; the four quadpairs tile 16x16.

A [BM, BN] accumulator lives as the 7-D tensor [MB, 2, 8, NB, 2, 2, 4] = (mb, mi3, mi2..0, nb, ni3, ni2, ni1..0),
m = 16 mb + mi, n = 16 nb + ni (row-major order, so a reshape to [BM, BN] moves nothing). In these coordinates the
A fragment depends only on (mb, mi3, mi2..0, ni1..0) and the B fragment only on (mi2..0, nb, ni3, ni1..0): each is
a plain broadcast of [MB, 2, 8, 1, 1, 1, 4] / [1, 1, 8, NB, 2, 1, 4], which `inline_asm_elementwise` broadcasts
itself, so one register fragment feeds every sub-tile that shares it. Operands come from shared memory laid out
[k-step x rows, 4] (k innermost, 8 bytes per row) and are read in a linear layout derived from the accumulator's;
the reshape/permute to the broadcast shape is a relabelling, which `convert_layout(assert_trivial=True)` proves.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

#: What the instruction below takes, as a hardware profile's `matrix_unit` names it (checked by
#: `_configs.matrix_unit`: a profile cannot declare dtypes this instruction does not compute in).
UNIT = {"shape": "m8n8k4", "operand_dtype": "float16", "accumulator_dtype": "float32"}

MMA_ASM: gl.constexpr = gl.constexpr(
    "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {$0,$1,$2,$3,$4,$5,$6,$7}, {$8, $9}, {$12, $13},"
    " {$16,$17,$18,$19,$20,$21,$22,$23};")
# pack=8: eight accumulator elements per call; f16 operands arrive two per 32-bit register, so A and B are four
# registers each of which the instruction reads the first two (k = 0..3; the other two repeat them).
MMA_CONSTRAINTS: gl.constexpr = gl.constexpr(",".join(["=f"] * 8 + ["r"] * 8 + ["f"] * 8))


def _bits(n: int):
    return [1 << i for i in range(n.bit_length() - 1)]


def _e7(d: int, x: int):
    v = [0] * 7
    v[d] = x
    return v


def acc_layout(MB: int, NB: int, WM: int, WN: int):
    """The [MB, 2, 8, NB, 2, 2, 4] accumulator over WM x WN warps (WM | MB, WN | NB, all powers of two)."""
    rm = [x * WM for x in _bits(MB // WM)]
    rn = [x * WN for x in _bits(NB // WN)]
    return gl.DistributedLinearLayout(
        reg_bases=[_e7(6, 1), _e7(2, 2), _e7(5, 1)] + [_e7(3, x) for x in rn] + [_e7(0, x) for x in rm],
        lane_bases=[_e7(2, 1), _e7(6, 2), _e7(1, 1), _e7(4, 1), _e7(2, 4)],
        warp_bases=[_e7(0, x) for x in _bits(WM)] + [_e7(3, x) for x in _bits(WN)],
        block_bases=[], shape=[MB, 2, 8, NB, 2, 2, 4])


def frag_a_layout(MB: int, NB: int, WM: int, WN: int):
    """One k-step of the A operand, [16 MB, 4] = (row, k), in the registers acc_layout(MB, NB, WM, WN) needs."""
    rm = [x * WM for x in _bits(MB // WM)]
    rn = [x * WN for x in _bits(NB // WN)]
    return gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 0]] + [[0, 0]] * len(rn) + [[16 * x, 0] for x in rm],
        lane_bases=[[1, 0], [2, 0], [8, 0], [0, 0], [4, 0]],
        warp_bases=[[16 * x, 0] for x in _bits(WM)] + [[0, 0]] * len(_bits(WN)),
        block_bases=[], shape=[16 * MB, 4])


def frag_b_layout(MB: int, NB: int, WM: int, WN: int):
    """One k-step of the B operand, [16 NB, 4] = (column, k), in the registers acc_layout(MB, NB, WM, WN) needs."""
    rm = [x * WM for x in _bits(MB // WM)]
    rn = [x * WN for x in _bits(NB // WN)]
    return gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 0]] + [[16 * x, 0] for x in rn] + [[0, 0]] * len(rm),
        lane_bases=[[1, 0], [2, 0], [0, 0], [8, 0], [4, 0]],
        warp_bases=[[0, 0]] * len(_bits(WM)) + [[16 * x, 0] for x in _bits(WN)],
        block_bases=[], shape=[16 * NB, 4])


@gluon.jit
def acc_rows_cols(L: gl.constexpr, MB: gl.constexpr, NB: gl.constexpr):
    """Row and column index of every accumulator element, broadcastable over L ([.., 1, 1, 1, 1] / [1, 1, 1, ..])."""
    o0 = gl.arange(0, MB, layout=gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, gl.SliceLayout(4, gl.SliceLayout(5, gl.SliceLayout(6, L)))))))
    o1 = gl.arange(0, 2, layout=gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, gl.SliceLayout(4, gl.SliceLayout(5, gl.SliceLayout(6, L)))))))
    o2 = gl.arange(0, 8, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, gl.SliceLayout(4, gl.SliceLayout(5, gl.SliceLayout(6, L)))))))
    o3 = gl.arange(0, NB, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(4, gl.SliceLayout(5, gl.SliceLayout(6, L)))))))
    o4 = gl.arange(0, 2, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, gl.SliceLayout(5, gl.SliceLayout(6, L)))))))
    o5 = gl.arange(0, 2, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, gl.SliceLayout(4, gl.SliceLayout(6, L)))))))
    o6 = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, gl.SliceLayout(4, gl.SliceLayout(5, L)))))))
    rows = o0[:, None, None, None, None, None, None] * 16 + o1[None, :, None, None, None, None, None] * 8 \
        + o2[None, None, :, None, None, None, None]
    cols = o3[None, None, None, :, None, None, None] * 16 + o4[None, None, None, None, :, None, None] * 8 \
        + o5[None, None, None, None, None, :, None] * 4 + o6[None, None, None, None, None, None, :]
    return rows, cols


@gluon.jit
def row_layout_rows(L: gl.constexpr, MB: gl.constexpr):
    """Row index [MB, 2, 8] in the layout of a row statistic of L (L with its four column dims reduced)."""
    RS: gl.constexpr = gl.SliceLayout(3, gl.SliceLayout(4, gl.SliceLayout(5, gl.SliceLayout(6, L))))
    r0 = gl.arange(0, MB, layout=gl.SliceLayout(1, gl.SliceLayout(2, RS)))
    r1 = gl.arange(0, 2, layout=gl.SliceLayout(0, gl.SliceLayout(2, RS)))
    r2 = gl.arange(0, 8, layout=gl.SliceLayout(0, gl.SliceLayout(1, RS)))
    return r0[:, None, None] * 16 + r1[None, :, None] * 8 + r2[None, None, :]


@gluon.jit
def row_max(x):
    return gl.max(gl.max(gl.max(gl.max(x, axis=6), axis=5), axis=4), axis=3)


@gluon.jit
def row_sum(x):
    return gl.sum(gl.sum(gl.sum(gl.sum(x, axis=6), axis=5), axis=4), axis=3)


@gluon.jit
def bcast_rows(r, L: gl.constexpr):
    """A [MB, 2, 8] row statistic -> broadcastable over accumulator L (any NB; the row bits are shared)."""
    RL: gl.constexpr = gl.SliceLayout(3, gl.SliceLayout(4, gl.SliceLayout(5, gl.SliceLayout(6, L))))
    r = gl.convert_layout(r, RL, assert_trivial=True)
    return gl.expand_dims(gl.expand_dims(gl.expand_dims(gl.expand_dims(r, 3), 4), 5), 6)


@gluon.jit
def stage_k_inner(smem, tile, R: gl.constexpr, KK: gl.constexpr):
    """tile [R, KK] whose k runs along its columns -> smem [(KK/4) R, 4] laid out [k-step, R, 4]."""
    smem.store(gl.reshape(gl.permute(gl.reshape(tile, [R, KK // 4, 4]), [1, 0, 2]), [(KK // 4) * R, 4]))


@gluon.jit
def stage_k_outer(smem, tile, KK: gl.constexpr, C: gl.constexpr):
    """tile [KK, C] whose k runs along its rows -> smem [(KK/4) C, 4] laid out [k-step, C, 4]. Conflict-free
    stores want each thread to hold 4 consecutive k of a column (a [4, x] size_per_thread load layout)."""
    smem.store(gl.reshape(gl.permute(gl.reshape(tile, [KK // 4, 4, C]), [0, 2, 1]), [(KK // 4) * C, 4]))


@gluon.jit
def mma_k4(sa, a_row0, sb, b_row0, acc, MB: gl.constexpr, NB: gl.constexpr,
           LA: gl.constexpr, LB: gl.constexpr, L: gl.constexpr):
    """acc += A[:, 4j:4j+4] @ B[4j:4j+4, :] for the k-step whose rows start at a_row0 / b_row0 of the staged
    operands (16 MB rows of sa, 16 NB rows of sb)."""
    fa = sa.slice(a_row0, 16 * MB).load(LA)                  # (mb, mi3, mi2, ni1, mi0 | mi1, ni0)
    fa = gl.reshape(fa, [MB, 2, 2, 2, 2, 2, 2])
    fa = gl.convert_layout(gl.reshape(gl.permute(fa, [0, 1, 2, 5, 4, 3, 6]), [MB, 2, 8, 1, 1, 1, 4]), L,
                           assert_trivial=True)
    fb = sb.slice(b_row0, 16 * NB).load(LB)                  # (nb, ni3, mi2, ni1, mi0 | mi1, ni0)
    fb = gl.reshape(fb, [NB, 2, 2, 2, 2, 2, 2])
    fb = gl.convert_layout(gl.reshape(gl.permute(fb, [2, 5, 4, 0, 1, 3, 6]), [1, 1, 8, NB, 2, 1, 4]), L,
                           assert_trivial=True)
    return gl.inline_asm_elementwise(MMA_ASM, MMA_CONSTRAINTS, [fa, fb, acc], dtype=gl.float32, is_pure=True,
                                     pack=8)
