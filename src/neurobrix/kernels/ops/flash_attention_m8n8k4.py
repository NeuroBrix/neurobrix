"""Flash-attention forward on the m8n8k4 matrix unit (sm_70 tensor cores) — our Gluon kernel.

The same contract as `flash_attention.py` (the wrapper's flash path): Q is scaled by `softmax_scale` and rounded to
its own dtype before QK^T, a memory-resident additive bias (the wrapper's "matrix" bias, stride 0 over the query
axis when there is no mask) is added to every score, softmax online, GQA by `GQA_GROUPS`, the log-sum-exp stored to
`Lse`, rows and keys of any length (tails masked), and a fully-masked row left to the wrapper's guard.

Built on `mma_m8n8k4`: one warp owns 16 query rows (the row statistics never cross a warp), S = Q K^T and O += P V
are HMMA, and Q (once per program), K, V and P (fp16) are staged through shared memory as [k-step x rows, 4].
The head dim is cut into at most two power-of-two parts DA + DB (the wrapper rounds it up to such a sum, the
extra columns loaded as zeros), so 96 runs as 64 + 32 and never as 128.
"""

import functools

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from .mma_m8n8k4 import (acc_rows_cols, bcast_rows, mma_k4, row_layout_rows, row_max, row_sum,
                         stage_k_inner, stage_k_outer)

LOG2E: gl.constexpr = gl.constexpr(1.4426950408889634)


@gluon.jit
def flash_attention_m8n8k4_kernel(
    Q, K, V, Bias, Out, Lse,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_bb, stride_bh, stride_bm,
    stride_ob, stride_oh, stride_om,
    nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim,
    MB: gl.constexpr, NS: gl.constexpr, DA: gl.constexpr, DB: gl.constexpr, GQA_GROUPS: gl.constexpr,
    LS: gl.constexpr, LOA: gl.constexpr, LOB: gl.constexpr,
    LAS: gl.constexpr, LAOA: gl.constexpr, LAOB: gl.constexpr,
    LBK: gl.constexpr, LBVA: gl.constexpr, LBVB: gl.constexpr,
    GR: gl.constexpr, GV: gl.constexpr,
):
    BM: gl.constexpr = 16 * MB
    BN: gl.constexpr = 16 * NS
    SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    m0 = gl.program_id(0).to(gl.int64) * BM
    off_hb = gl.program_id(1).to(gl.int64)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    off_h_kv = off_h // GQA_GROUPS
    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_h_kv * stride_kh
    v_base = V + off_b * stride_vb + off_h_kv * stride_vh
    b_base = Bias + off_b * stride_bb + off_h * stride_bh

    sq_a = gl.allocate_shared_memory(gl.float16, [(DA // 4) * BM, 4], SH)
    sk_a = gl.allocate_shared_memory(gl.float16, [(DA // 4) * BN, 4], SH)
    sv_a = gl.allocate_shared_memory(gl.float16, [(BN // 4) * DA, 4], SH)
    sp = gl.allocate_shared_memory(gl.float16, [(BN // 4) * BM, 4], SH)

    # Global tiles: GR reads rows along the head dim (Q, K), GV gives each thread 4 consecutive keys of a column (V).
    r_m = gl.arange(0, BM, layout=gl.SliceLayout(1, GR))[:, None]
    r_n = gl.arange(0, BN, layout=gl.SliceLayout(1, GR))[:, None]
    c_a = gl.arange(0, DA, layout=gl.SliceLayout(0, GR))[None, :]
    v_n = gl.arange(0, BN, layout=gl.SliceLayout(1, GV))[:, None]
    v_a = gl.arange(0, DA, layout=gl.SliceLayout(0, GV))[None, :]
    q_rows = m0 + r_m
    q_a = gl.load(q_base + q_rows * stride_qm + c_a, mask=(q_rows < seqlen_q) & (c_a < headdim), other=0.0)
    stage_k_inner(sq_a, (q_a * softmax_scale).to(gl.float16), BM, DA)
    acc_a = gl.zeros([MB, 2, 8, DA // 16, 2, 2, 4], gl.float32, layout=LOA)
    if DB > 0:
        sq_b = gl.allocate_shared_memory(gl.float16, [(DB // 4) * BM, 4], SH)
        sk_b = gl.allocate_shared_memory(gl.float16, [(DB // 4) * BN, 4], SH)
        sv_b = gl.allocate_shared_memory(gl.float16, [(BN // 4) * DB, 4], SH)
        c_b = DA + gl.arange(0, DB, layout=gl.SliceLayout(0, GR))[None, :]
        v_b = DA + gl.arange(0, DB, layout=gl.SliceLayout(0, GV))[None, :]
        q_b = gl.load(q_base + q_rows * stride_qm + c_b, mask=(q_rows < seqlen_q) & (c_b < headdim), other=0.0)
        stage_k_inner(sq_b, (q_b * softmax_scale).to(gl.float16), BM, DB)
        acc_b = gl.zeros([MB, 2, 8, DB // 16, 2, 2, 4], gl.float32, layout=LOB)

    s_rows, s_cols = acc_rows_cols(LS, MB, NS)
    s_rows = m0 + s_rows
    RS: gl.constexpr = gl.SliceLayout(3, gl.SliceLayout(4, gl.SliceLayout(5, gl.SliceLayout(6, LS))))
    m_i = gl.full([MB, 2, 8], float("-inf"), gl.float32, layout=RS)      # running max, log2 domain
    l_i = gl.zeros([MB, 2, 8], gl.float32, layout=RS)

    for n0 in range(0, seqlen_k, BN):
        gl.barrier()                                       # every warp is done reading the previous tile
        k_rows = n0 + r_n
        v_rows = n0 + v_n
        stage_k_inner(sk_a, gl.load(k_base + k_rows * stride_kn + c_a,
                                    mask=(k_rows < seqlen_k) & (c_a < headdim), other=0.0), BN, DA)
        stage_k_outer(sv_a, gl.load(v_base + v_rows * stride_vn + v_a,
                                    mask=(v_rows < seqlen_k) & (v_a < headdim), other=0.0), BN, DA)
        if DB > 0:
            stage_k_inner(sk_b, gl.load(k_base + k_rows * stride_kn + c_b,
                                        mask=(k_rows < seqlen_k) & (c_b < headdim), other=0.0), BN, DB)
            stage_k_outer(sv_b, gl.load(v_base + v_rows * stride_vn + v_b,
                                        mask=(v_rows < seqlen_k) & (v_b < headdim), other=0.0), BN, DB)
        gl.barrier()
        s = gl.zeros([MB, 2, 8, NS, 2, 2, 4], gl.float32, layout=LS)
        for j in gl.static_range(DA // 4):
            s = mma_k4(sq_a, j * BM, sk_a, j * BN, s, MB, NS, LAS, LBK, LS)
        if DB > 0:
            for j in gl.static_range(DB // 4):
                s = mma_k4(sq_b, j * BM, sk_b, j * BN, s, MB, NS, LAS, LBK, LS)
        cols = n0 + s_cols
        bias = gl.load(b_base + s_rows * stride_bm + cols, mask=(s_rows < seqlen_q) & (cols < seqlen_k),
                       other=0.0).to(gl.float32)
        s = gl.where(cols < seqlen_k, (s + bias) * LOG2E, float("-inf"))
        m_new = gl.maximum(m_i, row_max(s))
        p = gl.exp2(s - bcast_rows(m_new, LS))
        alpha = gl.exp2(m_i - m_new)
        l_i = l_i * alpha + row_sum(p)
        m_i = m_new
        stage_k_inner(sp, gl.reshape(p.to(gl.float16), [BM, BN]), BM, BN)
        acc_a = acc_a * bcast_rows(alpha, LOA)
        if DB > 0:
            acc_b = acc_b * bcast_rows(alpha, LOB)
        gl.barrier()
        for j in gl.static_range(BN // 4):
            acc_a = mma_k4(sp, j * BM, sv_a, j * DA, acc_a, MB, DA // 16, LAOA, LBVA, LOA)
            if DB > 0:
                acc_b = mma_k4(sp, j * BM, sv_b, j * DB, acc_b, MB, DB // 16, LAOB, LBVB, LOB)

    lse_rows = m0 + row_layout_rows(LS, MB)
    gl.store(Lse + off_hb * seqlen_q_rounded + lse_rows, (m_i + gl.log2(l_i)) / LOG2E,
             mask=lse_rows < seqlen_q_rounded)
    o_base = Out + off_b * stride_ob + off_h * stride_oh
    o_rows, o_cols = acc_rows_cols(LOA, MB, DA // 16)
    o_rows = m0 + o_rows
    gl.store(o_base + o_rows * stride_om + o_cols, (acc_a / bcast_rows(l_i, LOA)).to(Out.dtype.element_ty),
             mask=(o_rows < seqlen_q) & (o_cols < headdim))
    if DB > 0:
        o_rows_b, o_cols_b = acc_rows_cols(LOB, MB, DB // 16)
        o_rows_b = m0 + o_rows_b
        o_cols_b = DA + o_cols_b
        gl.store(o_base + o_rows_b * stride_om + o_cols_b, (acc_b / bcast_rows(l_i, LOB)).to(Out.dtype.element_ty),
                 mask=(o_rows_b < seqlen_q) & (o_cols_b < headdim))


def head_parts(headdim: int):
    """The head dim as DA + DB, each a power of two >= 16 (DB may be 0): the smallest such sum >= headdim."""
    d = -(-headdim // 16) * 16
    while bin(d // 16).count("1") > 2:
        d += 16
    da = 1 << (d.bit_length() - 1)
    return da, d - da


@functools.lru_cache(maxsize=None)
def flash_layouts(MB: int, NS: int, DA: int, DB: int) -> dict:
    """Every layout the kernel takes, for one warp per 16-row band (MB warps, WN = 1)."""
    from .mma_m8n8k4 import acc_layout, frag_a_layout, frag_b_layout
    na, nb = DA // 16, max(DB // 16, 1)
    return dict(
        LS=acc_layout(MB, NS, MB, 1), LOA=acc_layout(MB, na, MB, 1), LOB=acc_layout(MB, nb, MB, 1),
        LAS=frag_a_layout(MB, NS, MB, 1), LAOA=frag_a_layout(MB, na, MB, 1), LAOB=frag_a_layout(MB, nb, MB, 1),
        LBK=frag_b_layout(MB, NS, MB, 1), LBVA=frag_b_layout(MB, na, MB, 1), LBVB=frag_b_layout(MB, nb, MB, 1),
        GR=gl.BlockedLayout([1, 8], [8, 4], [MB, 1], [1, 0]),
        GV=gl.BlockedLayout([4, 2], [1, 32], [MB, 1], [1, 0]))
