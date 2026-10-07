"""(Measurement record, kept as run; the maintained form is src/neurobrix/kernels/ops/mma_m8n8k4.py and flash_attention_m8n8k4.py.)
Step 2 v2: the tensor-core matmul block with coalesced global loads staged through shared memory.
Accumulator: 7-D [MB, 2, 8, NB, 2, 2, 4] = (mb, mi3, mi2..0, nb, ni3, ni2, ni1..0), m = 16 mb + mi, n = 16 nb + ni,
holding the m8n8k4 fragment of step 1. In these coordinates A's fragment is a broadcast of [MB, 2, 8, 1, 1, 1, 4]
and B's of [1, 1, 8, NB, 2, 1, 4]. Shared memory holds A as [J, BM, 4] and B as [J, BN, 4] (k innermost, 8 bytes per
row): one k-step is .index(j), read in a linear layout derived from the accumulator's, then reshaped/permuted in
registers to the broadcast operand; convert_layout(assert_trivial=True) proves that relabelling moves no data."""
import os, time
import numpy as np
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator, NBXDtype

ASM: gl.constexpr = gl.constexpr(
    "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {$0,$1,$2,$3,$4,$5,$6,$7}, {$8, $9}, {$12, $13},"
    " {$16,$17,$18,$19,$20,$21,$22,$23};")
CONS: gl.constexpr = gl.constexpr(",".join(["=f"] * 8 + ["r"] * 8 + ["f"] * 8))


def _bits(n):
    return [1 << i for i in range(n.bit_length() - 1)]


def layouts(MB, NB, WM, WN, BK, nw):
    wm, wn = _bits(WM), _bits(WN)
    rm, rn = [x * WM for x in _bits(MB // WM)], [x * WN for x in _bits(NB // WN)]
    z7 = [0] * 7
    def e(d, x):
        v = list(z7); v[d] = x; return v
    L = gl.DistributedLinearLayout(
        reg_bases=[e(6, 1), e(2, 2), e(5, 1)] + [e(3, x) for x in rn] + [e(0, x) for x in rm],
        lane_bases=[e(2, 1), e(6, 2), e(1, 1), e(4, 1), e(2, 4)],
        warp_bases=[e(0, x) for x in wm] + [e(3, x) for x in wn], block_bases=[], shape=[MB, 2, 8, NB, 2, 2, 4])
    LA = gl.DistributedLinearLayout(   # over [BM, 4] = (row, k)
        reg_bases=[[0, 1], [0, 2], [0, 0]] + [[0, 0]] * len(rn) + [[16 * x, 0] for x in rm],
        lane_bases=[[1, 0], [2, 0], [8, 0], [0, 0], [4, 0]],
        warp_bases=[[16 * x, 0] for x in wm] + [[0, 0]] * len(wn), block_bases=[], shape=[16 * MB, 4])
    LB = gl.DistributedLinearLayout(   # over [BN, 4] = (col, k)
        reg_bases=[[0, 1], [0, 2], [0, 0]] + [[16 * x, 0] for x in rn] + [[0, 0]] * len(rm),
        lane_bases=[[1, 0], [2, 0], [0, 0], [8, 0], [4, 0]],
        warp_bases=[[0, 0]] * len(wm) + [[16 * x, 0] for x in wn], block_bases=[], shape=[16 * NB, 4])
    # coalesced global tiles: A [BM, BK] contiguous along k, B [BK, BN] contiguous along n, 8 halves per thread
    GA = gl.BlockedLayout([1, 8], [32 // (BK // 8), BK // 8], [nw, 1], [1, 0])
    GB = gl.BlockedLayout([4, 2], [1, 32], [nw, 1], [1, 0])   # 4 consecutive k per thread: STS.64 into [n, k]
    idx = []
    for d in range(7):
        S = L
        for i in reversed(range(7)):
            if i != d:
                S = gl.SliceLayout(i, S)
        idx.append(S)
    return L, LA, LB, GA, GB, idx


@gluon.jit
def mm_smem(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn,
            MB: gl.constexpr, NB: gl.constexpr, BK: gl.constexpr,
            L: gl.constexpr, LA: gl.constexpr, LB: gl.constexpr, GA: gl.constexpr, GB: gl.constexpr,
            I0: gl.constexpr, I1: gl.constexpr, I2: gl.constexpr, I3: gl.constexpr, I4: gl.constexpr,
            I5: gl.constexpr, I6: gl.constexpr):
    BM: gl.constexpr = 16 * MB
    BN: gl.constexpr = 16 * NB
    J: gl.constexpr = BK // 4
    m0 = gl.program_id(0) * BM
    n0 = gl.program_id(1) * BN
    SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    sa = gl.allocate_shared_memory(gl.float16, [J * BM, 4], SH)
    sb = gl.allocate_shared_memory(gl.float16, [J * BN, 4], SH)
    ga_r = gl.arange(0, BM, layout=gl.SliceLayout(1, GA))[:, None]
    ga_k = gl.arange(0, BK, layout=gl.SliceLayout(0, GA))[None, :]
    gb_k = gl.arange(0, BK, layout=gl.SliceLayout(1, GB))[:, None]
    gb_c = gl.arange(0, BN, layout=gl.SliceLayout(0, GB))[None, :]
    pa = A + (m0 + ga_r) * sam + ga_k * sak
    pb = B + gb_k * sbk + (n0 + gb_c) * sbn
    acc = gl.zeros([MB, 2, 8, NB, 2, 2, 4], gl.float32, layout=L)
    ta = gl.load(pa)
    tb = gl.load(pb)
    for k0 in range(0, K, BK):
        gl.barrier()
        sa.store(gl.reshape(gl.permute(gl.reshape(ta, [BM, J, 4]), [1, 0, 2]), [J * BM, 4]))
        sb.store(gl.reshape(gl.permute(gl.reshape(tb, [J, 4, BN]), [0, 2, 1]), [J * BN, 4]))
        gl.barrier()
        if k0 + BK < K:                                   # prefetch the next tile while the HMMAs run
            ta = gl.load(pa + (k0 + BK) * sak)
            tb = gl.load(pb + (k0 + BK) * sbk)
        for j in gl.static_range(J):
            fa = sa.slice(j * BM, BM).load(LA)                     # [BM, 4] = (mb, mi3, mi2, ni1, mi0 | mi1, ni0)
            fa = gl.reshape(fa, [MB, 2, 2, 2, 2, 2, 2])
            fa = gl.reshape(gl.permute(fa, [0, 1, 2, 5, 4, 3, 6]), [MB, 2, 8, 1, 1, 1, 4])
            fa = gl.convert_layout(fa, L, assert_trivial=True)
            fb = sb.slice(j * BN, BN).load(LB)                     # [BN, 4] = (nb, ni3, mi2, ni1, mi0 | mi1, ni0)
            fb = gl.reshape(fb, [NB, 2, 2, 2, 2, 2, 2])
            fb = gl.reshape(gl.permute(fb, [2, 5, 4, 0, 1, 3, 6]), [1, 1, 8, NB, 2, 1, 4])
            fb = gl.convert_layout(fb, L, assert_trivial=True)
            acc = gl.inline_asm_elementwise(ASM, CONS, [fa, fb, acc], dtype=gl.float32, is_pure=True, pack=8)
    o0 = gl.arange(0, MB, layout=I0)[:, None, None, None, None, None, None]
    o1 = gl.arange(0, 2, layout=I1)[None, :, None, None, None, None, None]
    o2 = gl.arange(0, 8, layout=I2)[None, None, :, None, None, None, None]
    o3 = gl.arange(0, NB, layout=I3)[None, None, None, :, None, None, None]
    o4 = gl.arange(0, 2, layout=I4)[None, None, None, None, :, None, None]
    o5 = gl.arange(0, 2, layout=I5)[None, None, None, None, None, :, None]
    o6 = gl.arange(0, 4, layout=I6)[None, None, None, None, None, None, :]
    rows = m0 + o0 * 16 + o1 * 8 + o2
    cols = n0 + o3 * 16 + o4 * 8 + o5 * 4 + o6
    gl.store(C + rows * scm + cols * scn, acc)


def run(a, b, MB, NB, WM, WN, BK):
    M, K = a.shape; N = b.shape[1]; nw = WM * WN
    L, LA, LB, GA, GB, idx = layouts(MB, NB, WM, WN, BK, nw)
    c = NBXTensor.empty((M, N), NBXDtype.float32, "cuda:0")
    mm_smem[(M // (16 * MB), N // (16 * NB))](a, b, c, M, N, K, *a.stride(), *b.stride(), *c.stride(),
                                             MB=MB, NB=NB, BK=BK, L=L, LA=LA, LB=LB, GA=GA, GB=GB,
                                             I0=idx[0], I1=idx[1], I2=idx[2], I3=idx[3], I4=idx[4], I5=idx[5],
                                             I6=idx[6], num_warps=nw)
    return c


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    cfgs = [tuple(int(x) for x in c.split(",")) for c in os.environ.get("CFG", "8,8,2,2,32").split(";")]
    M = N = K = 512
    an = rng.standard_normal((M, K)).astype(np.float16); bn = rng.standard_normal((K, N)).astype(np.float16)
    ref = an.astype(np.float64) @ bn.astype(np.float64)
    ai = rng.integers(-3, 4, (M, K)).astype(np.float16); bi = rng.integers(-3, 4, (K, N)).astype(np.float16)
    refi = ai.astype(np.float64) @ bi.astype(np.float64)
    a = NBXTensor.from_numpy(an).to("cuda:0"); b = NBXTensor.from_numpy(bn).to("cuda:0")
    a_i = NBXTensor.from_numpy(ai).to("cuda:0"); b_i = NBXTensor.from_numpy(bi).to("cuda:0")
    S = int(os.environ.get("SIZE", "4096"))
    al = NBXTensor.from_numpy(rng.standard_normal((S, S)).astype(np.float16)).to("cuda:0")
    bl = NBXTensor.from_numpy(rng.standard_normal((S, S)).astype(np.float16)).to("cuda:0")
    for MB, NB, WM, WN, BK in cfgs:
        tag = f"block {16*MB}x{16*NB}x{BK} warps {WM}x{WN}"
        try:
            ci = run(a_i, b_i, MB, NB, WM, WN, BK); c = run(a, b, MB, NB, WM, WN, BK); DeviceAllocator.sync_device()
            ei = np.abs(ci.numpy() - refi).max()
            err = np.abs(c.numpy() - ref).max() / np.abs(ref).max()
            run(al, bl, MB, NB, WM, WN, BK); DeviceAllocator.sync_device()
            t0 = time.perf_counter(); n = 5
            for _ in range(n):
                run(al, bl, MB, NB, WM, WN, BK)
            DeviceAllocator.sync_device(); dt = (time.perf_counter() - t0) / n
            print(f"{tag}: int max err {ei:.1f} rel err {err:.2e}  {S}^3 {dt*1e3:.2f} ms "
                  f"{2*S**3/dt/1e12:.2f} TFLOP/s", flush=True)
        except Exception as ex:
            print(f"{tag}: ERROR {type(ex).__name__} {str(ex)[-900:]}", flush=True)
