"""(Measurement record, kept as run; the maintained form is src/neurobrix/kernels/ops/mma_m8n8k4.py and flash_attention_m8n8k4.py.)
Step 3 (inbox 2026-10-07 19:23): flash-attention forward on Volta tensor cores, built from the step-2 block.
One warp owns 16 query rows (no cross-warp softmax). S = Q K^T and O += P V are m8n8k4 HMMA in the 7-D fragment
coordinates of step 2; Q (once), K, V and P (fp16) are staged through shared memory as [k-step x rows, 4]. The head
dim is split into two powers of two (96 = 64 + 32), never padded. Masks: query and key tails (T need not divide)."""
import os, sys, time
import numpy as np
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator, NBXDtype

ASM: gl.constexpr = gl.constexpr(
    "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {$0,$1,$2,$3,$4,$5,$6,$7}, {$8, $9}, {$12, $13},"
    " {$16,$17,$18,$19,$20,$21,$22,$23};")
CONS: gl.constexpr = gl.constexpr(",".join(["=f"] * 8 + ["r"] * 8 + ["f"] * 8))
LOG2E: gl.constexpr = gl.constexpr(1.4426950408889634)


def _bits(n):
    return [1 << i for i in range(n.bit_length() - 1)]


def acc_layout(MB, NB, nw):
    """Accumulator [MB,2,8,NB,2,2,4]; one warp per 16-row band (MB == nw)."""
    rn = _bits(NB)
    def e(d, x):
        v = [0] * 7; v[d] = x; return v
    return gl.DistributedLinearLayout(
        reg_bases=[e(6, 1), e(2, 2), e(5, 1)] + [e(3, x) for x in rn],
        lane_bases=[e(2, 1), e(6, 2), e(1, 1), e(4, 1), e(2, 4)],
        warp_bases=[e(0, x) for x in _bits(nw)], block_bases=[], shape=[MB, 2, 8, NB, 2, 2, 4])


def frag_a(MB, NB, nw):     # A operand over [16 MB, 4] = (row, k) for an accumulator with NB column blocks
    return gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 0]] + [[0, 0]] * len(_bits(NB)),
        lane_bases=[[1, 0], [2, 0], [8, 0], [0, 0], [4, 0]],
        warp_bases=[[16 * x, 0] for x in _bits(nw)], block_bases=[], shape=[16 * MB, 4])


def frag_b(MB, NB, nw):     # B operand over [16 NB, 4] = (col, k)
    return gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 0]] + [[16 * x, 0] for x in _bits(NB)],
        lane_bases=[[1, 0], [2, 0], [0, 0], [8, 0], [4, 0]],
        warp_bases=[[0, 0]] * len(_bits(nw)), block_bases=[], shape=[16 * NB, 4])


@gluon.jit
def _rows_cols(L: gl.constexpr, MB: gl.constexpr, NB: gl.constexpr):
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
def _rowmax(x):
    return gl.max(gl.max(gl.max(gl.max(x, axis=6), axis=5), axis=4), axis=3)


@gluon.jit
def _rowsum(x):
    return gl.sum(gl.sum(gl.sum(gl.sum(x, axis=6), axis=5), axis=4), axis=3)


@gluon.jit
def _bcast(r, L: gl.constexpr):     # [MB,2,8] row statistic -> broadcastable over accumulator L
    RL: gl.constexpr = gl.SliceLayout(3, gl.SliceLayout(4, gl.SliceLayout(5, gl.SliceLayout(6, L))))
    r = gl.convert_layout(r, RL, assert_trivial=True)
    return gl.expand_dims(gl.expand_dims(gl.expand_dims(gl.expand_dims(r, 3), 4), 5), 6)


@gluon.jit
def _stage_rows(smem, tile, R: gl.constexpr, DD: gl.constexpr, off: gl.constexpr):
    """tile [R, DD] (k = DD innermost) -> smem rows [off + (DD/4) R, 4] laid out [k-step, R, 4]."""
    smem.slice(off, (DD // 4) * R).store(
        gl.reshape(gl.permute(gl.reshape(tile, [R, DD // 4, 4]), [1, 0, 2]), [(DD // 4) * R, 4]))


@gluon.jit
def _stage_cols(smem, tile, R: gl.constexpr, DD: gl.constexpr, off: gl.constexpr):
    """tile [R, DD] with k = R (rows) -> smem [off + (R/4) DD, 4] laid out [k-step, DD, 4]."""
    smem.slice(off, (R // 4) * DD).store(
        gl.reshape(gl.permute(gl.reshape(tile, [R // 4, 4, DD]), [0, 2, 1]), [(R // 4) * DD, 4]))


@gluon.jit
def _mma_step(sa, a_off, sb, b_off, acc, MB: gl.constexpr, NB: gl.constexpr,
              LA: gl.constexpr, LB: gl.constexpr, L: gl.constexpr):
    fa = sa.slice(a_off, 16 * MB).load(LA)
    fa = gl.reshape(fa, [MB, 2, 2, 2, 2, 2, 2])
    fa = gl.convert_layout(gl.reshape(gl.permute(fa, [0, 1, 2, 5, 4, 3, 6]), [MB, 2, 8, 1, 1, 1, 4]), L, assert_trivial=True)
    fb = sb.slice(b_off, 16 * NB).load(LB)
    fb = gl.reshape(fb, [NB, 2, 2, 2, 2, 2, 2])
    fb = gl.convert_layout(gl.reshape(gl.permute(fb, [2, 5, 4, 0, 1, 3, 6]), [1, 1, 8, NB, 2, 1, 4]), L, assert_trivial=True)
    return gl.inline_asm_elementwise(ASM, CONS, [fa, fb, acc], dtype=gl.float32, is_pure=True, pack=8)


@gluon.jit
def fa_tc(Q, K, V, O, sm_scale, sqb, sqh, sqm, skb, skh, skn, svb, svh, svn, sob, soh, som, H, T,
          MB: gl.constexpr, NS: gl.constexpr, DA: gl.constexpr, DB: gl.constexpr,
          LS: gl.constexpr, LOA: gl.constexpr, LOB: gl.constexpr,
          LAS: gl.constexpr, LAOA: gl.constexpr, LAOB: gl.constexpr, LBK: gl.constexpr, LBVA: gl.constexpr,
          LBVB: gl.constexpr, GR: gl.constexpr, GV: gl.constexpr):
    BM: gl.constexpr = 16 * MB
    BN: gl.constexpr = 16 * NS
    DQ: gl.constexpr = DA + DB
    pid_m = gl.program_id(0)
    bh = gl.program_id(1)
    b = bh // H
    h = bh % H
    m0 = pid_m * BM
    SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    sq = gl.allocate_shared_memory(gl.float16, [(DA // 4) * BM, 4], SH)   # power-of-two buffers per head-dim part
    sk = gl.allocate_shared_memory(gl.float16, [(DA // 4) * BN, 4], SH)
    sp = gl.allocate_shared_memory(gl.float16, [(BN // 4) * BM, 4], SH)
    sva = gl.allocate_shared_memory(gl.float16, [(BN // 4) * DA, 4], SH)
    qbase = Q + b * sqb + h * sqh
    kbase = K + b * skb + h * skh
    vbase = V + b * svb + h * svh
    # global index tiles
    r_m = gl.arange(0, BM, layout=gl.SliceLayout(1, GR))[:, None]
    r_n = gl.arange(0, BN, layout=gl.SliceLayout(1, GR))[:, None]
    c_a = gl.arange(0, DA, layout=gl.SliceLayout(0, GR))[None, :]
    v_n = gl.arange(0, BN, layout=gl.SliceLayout(1, GV))[:, None]
    v_a = gl.arange(0, DA, layout=gl.SliceLayout(0, GV))[None, :]
    qa = gl.load(qbase + (m0 + r_m) * sqm + c_a, mask=(m0 + r_m) < T, other=0.0)
    _stage_rows(sq, qa, BM, DA, 0)
    if DB > 0:
        svb_s = gl.allocate_shared_memory(gl.float16, [(BN // 4) * DB, 4], SH)
        sqb_s = gl.allocate_shared_memory(gl.float16, [(DB // 4) * BM, 4], SH)
        skb_s = gl.allocate_shared_memory(gl.float16, [(DB // 4) * BN, 4], SH)
        c_b = gl.arange(0, DB, layout=gl.SliceLayout(0, GR))[None, :]
        v_b = gl.arange(0, DB, layout=gl.SliceLayout(0, GV))[None, :]
        qb = gl.load(qbase + (m0 + r_m) * sqm + DA + c_b, mask=(m0 + r_m) < T, other=0.0)
        _stage_rows(sqb_s, qb, BM, DB, 0)
        acc_b = gl.zeros([MB, 2, 8, DB // 16, 2, 2, 4], gl.float32, layout=LOB)
    acc_a = gl.zeros([MB, 2, 8, DA // 16, 2, 2, 4], gl.float32, layout=LOA)
    srows, scols = _rows_cols(LS, MB, NS)
    RS: gl.constexpr = gl.SliceLayout(3, gl.SliceLayout(4, gl.SliceLayout(5, gl.SliceLayout(6, LS))))
    m_i = gl.full([MB, 2, 8], float("-inf"), gl.float32, layout=RS)
    l_i = gl.zeros([MB, 2, 8], gl.float32, layout=RS)
    qk_scale = sm_scale * LOG2E
    for n0 in range(0, T, BN):
        gl.barrier()                                         # previous tile's readers are done
        ka = gl.load(kbase + (n0 + r_n) * skn + c_a, mask=(n0 + r_n) < T, other=0.0)
        _stage_rows(sk, ka, BN, DA, 0)
        va = gl.load(vbase + (n0 + v_n) * svn + v_a, mask=(n0 + v_n) < T, other=0.0)
        _stage_cols(sva, va, BN, DA, 0)
        if DB > 0:
            kb = gl.load(kbase + (n0 + r_n) * skn + DA + c_b, mask=(n0 + r_n) < T, other=0.0)
            _stage_rows(skb_s, kb, BN, DB, 0)
            vb = gl.load(vbase + (n0 + v_n) * svn + DA + v_b, mask=(n0 + v_n) < T, other=0.0)
            _stage_cols(svb_s, vb, BN, DB, 0)
        gl.barrier()
        s = gl.zeros([MB, 2, 8, NS, 2, 2, 4], gl.float32, layout=LS)
        for j in gl.static_range(DA // 4):
            s = _mma_step(sq, j * BM, sk, j * BN, s, MB, NS, LAS, LBK, LS)
        if DB > 0:
            for j in gl.static_range(DB // 4):
                s = _mma_step(sqb_s, j * BM, skb_s, j * BN, s, MB, NS, LAS, LBK, LS)
        s = gl.where(n0 + scols < T, s * qk_scale, float("-inf"))
        m_new = gl.maximum(m_i, _rowmax(s))
        p = gl.exp2(s - _bcast(m_new, LS))
        alpha = gl.exp2(m_i - m_new)
        l_i = l_i * alpha + _rowsum(p)
        m_i = m_new
        acc_a = acc_a * _bcast(alpha, LOA)
        _stage_rows(sp, gl.reshape(p.to(gl.float16), [BM, BN]), BM, BN, 0)
        if DB > 0:
            acc_b = acc_b * _bcast(alpha, LOB)
        gl.barrier()
        for j in gl.static_range(BN // 4):
            acc_a = _mma_step(sp, j * BM, sva, j * DA, acc_a, MB, DA // 16, LAOA, LBVA, LOA)
            if DB > 0:
                acc_b = _mma_step(sp, j * BM, svb_s, j * DB, acc_b, MB, DB // 16, LAOB, LBVB, LOB)
    orow, ocol = _rows_cols(LOA, MB, DA // 16)
    obase = O + b * sob + h * soh
    gl.store(obase + (m0 + orow) * som + ocol, (acc_a / _bcast(l_i, LOA)).to(gl.float16), mask=(m0 + orow) < T)
    if DB > 0:
        orow2, ocol2 = _rows_cols(LOB, MB, DB // 16)
        gl.store(obase + (m0 + orow2) * som + DA + ocol2, (acc_b / _bcast(l_i, LOB)).to(gl.float16),
                 mask=(m0 + orow2) < T)


def run(q, k, v, MB, NS):
    B, H, T, D = q.shape
    DA = 1 << (D.bit_length() - 1); DB = D - DA
    assert DB == 0 or (DB & (DB - 1) == 0 and DB >= 16), D
    nw = MB
    o = NBXTensor.empty((B, H, T, D), NBXDtype.float16, "cuda:0")
    NA, NBb = DA // 16, max(DB // 16, 1)
    fa_tc[((T + 16 * MB - 1) // (16 * MB), B * H)](
        q, k, v, o, 1.0 / D ** 0.5, *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], *o.stride()[:3], H, T,
        MB=MB, NS=NS, DA=DA, DB=DB, LS=acc_layout(MB, NS, nw), LOA=acc_layout(MB, NA, nw),
        LOB=acc_layout(MB, NBb, nw), LAS=frag_a(MB, NS, nw), LAOA=frag_a(MB, NA, nw), LAOB=frag_a(MB, NBb, nw),
        LBK=frag_b(MB, NS, nw), LBVA=frag_b(MB, NA, nw), LBVB=frag_b(MB, NBb, nw),
        GR=gl.BlockedLayout([1, 8], [8, 4], [nw, 1], [1, 0]), GV=gl.BlockedLayout([4, 2], [1, 32], [nw, 1], [1, 0]),
        num_warps=nw)
    return o


if __name__ == "__main__":
    from neurobrix.kernels import wrappers as W
    B, H, T, D = [int(x) for x in sys.argv[1:5]]
    cfgs = [tuple(int(x) for x in c.split(",")) for c in os.environ.get("CFG", "4,4").split(";")]
    rng = np.random.default_rng(0)
    mk = lambda: NBXTensor.from_numpy((rng.standard_normal((B, H, T, D)) * 0.5).astype(np.float16)).to("cuda:0")
    q, k, v = mk(), mk(), mk()
    fl = 4.0 * B * H * T * T * D
    if T <= 4096:   # fp64 oracle
        qn, kn, vn = (x.numpy().astype(np.float64) for x in (q, k, v))
        s = qn @ kn.transpose(0, 1, 3, 2) / D ** 0.5
        p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
        r = (p @ vn).astype(np.float32); rname = "fp64"
    else:           # the engine's FMA flash kernel as oracle (proven 10-07 against the container oracle)
        W._lk.sdpa_route = lambda *a, **kw: ("flash", 0)
        ref = W.scaled_dot_product_attention_wrapper(q, k, v, k_pre_transposed=False); DeviceAllocator.sync_device()
        t0 = time.perf_counter(); ref = W.scaled_dot_product_attention_wrapper(q, k, v, k_pre_transposed=False)
        DeviceAllocator.sync_device(); t_ref = time.perf_counter() - t0
        print(f"engine flash (FMA) T{T} D{D}: {t_ref*1e3:.0f} ms {fl/t_ref/1e12:.2f} TFLOP/s", flush=True)
        r = ref.numpy().astype(np.float32); rname = "engine"
    for MB, NS in cfgs:
        tag = f"tc BM{16*MB} BN{16*NS} w{MB}"
        try:
            run(q, k, v, MB, NS); DeviceAllocator.sync_device()
            t0 = time.perf_counter(); o = run(q, k, v, MB, NS); DeviceAllocator.sync_device()
            dt = time.perf_counter() - t0
            print(f"{tag}: {dt*1e3:.1f} ms {fl/dt/1e12:.2f} TFLOP/s max|diff| vs {rname} "
                  f"{np.abs(o.numpy().astype(np.float32) - r).max():.2e}", flush=True)
        except Exception as e:
            print(f"{tag}: ERROR {type(e).__name__} {str(e)[-1200:]} CAUSE {repr(e.__cause__)[:600]}", flush=True)
