"""(Measurement record, kept as run; the maintained form is src/neurobrix/kernels/ops/mma_m8n8k4.py and flash_attention_m8n8k4.py.)
Step 1 (inbox 2026-10-07 19:23): one 16x16x4 tile D = A @ B + C on Volta tensor cores, from our own Gluon kernel.
One warp = four quadpairs (lanes {a + 16 b} + 4 q); quadpair q computes the 8x8 quadrant (q % 2, q // 2) with
mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 emitted by inline asm. Fragment placement (CUTLASS
cute/atom/mma_traits_sm70.hpp, SM70_8x8x4_F32F16F16F32_TN: ALayout = BLayout = SM70_8x4_Row, CLayout = SM70_8x8_32b):
thread t = a0 + 2 a1 + 4 b holds row t of A and column t of B (k = 0..3 in its registers 0..3); its accumulator
register v = v0 + 2 v1 + 4 v2 holds D[m = a0 + 2 v1 + 4 b, n = v0 + 2 a1 + 4 v2]. The layout below states exactly
that (register bits, lane bits) -> (m, n) map; every operand is carried in it so each asm call sees one thread's
8 registers in order."""
import sys
import numpy as np
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator

# Triton hands 16-bit operands two per 32-bit register: A and B arrive as 4 "r" each, ($8, $9) = (a0 a1, a2 a3).
ASM: gl.constexpr = gl.constexpr(
    "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {$0,$1,$2,$3,$4,$5,$6,$7}, {$8, $9}, {$12, $13},"
    " {$16,$17,$18,$19,$20,$21,$22,$23};")
CONS: gl.constexpr = gl.constexpr(",".join(["=f"] * 8 + ["r"] * 8 + ["f"] * 8))


@gluon.jit
def tile16(A, B, C, Dout):
    # (reg bits v0 v1 v2, lane bits a0 a1 q0 q1 b) -> (m, n) of the 16x16 tile
    L: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [2, 0], [0, 4]],
        lane_bases=[[1, 0], [0, 2], [8, 0], [0, 8], [4, 0]],
        warp_bases=[], block_bases=[], shape=[16, 16])
    m = gl.arange(0, 16, layout=gl.SliceLayout(1, L))[:, None]
    n = gl.arange(0, 16, layout=gl.SliceLayout(0, L))[None, :]
    mm = m % 8
    nn = n % 8
    # thread index t inside the quadpair and register index v, recovered from (m, n) by the same bijection
    t = (mm % 2) + 2 * ((nn // 2) % 2) + 4 * (mm // 4)
    v = (nn % 2) + 2 * ((mm // 2) % 2) + 4 * (nn // 4)
    k = v % 4
    a = gl.load(A + ((m // 8) * 8 + t) * 4 + k)          # A is 16x4 row-major: row 8 qm + t, column k
    b = gl.load(B + k * 16 + (n // 8) * 8 + t)          # B is 4x16 row-major: row k, column 8 qn + t
    c = gl.load(C + m * 16 + n)
    d = gl.inline_asm_elementwise(ASM, CONS, [a, b, c], dtype=gl.float32, is_pure=True, pack=8)
    gl.store(Dout + m * 16 + n, d)


def check(a, b, c):
    A = NBXTensor.from_numpy(a).to("cuda:0"); B = NBXTensor.from_numpy(b).to("cuda:0")
    C = NBXTensor.from_numpy(c).to("cuda:0"); Dt = NBXTensor.from_numpy(np.zeros((16, 16), np.float32)).to("cuda:0")
    tile16[(1,)](A, B, C, Dt, num_warps=1)
    DeviceAllocator.sync_device()
    return Dt.numpy(), a.astype(np.float64) @ b.astype(np.float64) + c


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # exact: small integers, every product and partial sum representable -> D must equal the oracle bit for bit
    a = rng.integers(-8, 8, (16, 4)).astype(np.float16); b = rng.integers(-8, 8, (4, 16)).astype(np.float16)
    c = rng.integers(-64, 64, (16, 16)).astype(np.float32)
    d, ref = check(a, b, c)
    print(f"integers: max|D - oracle| = {np.abs(d - ref).max()}  exact={np.array_equal(d, ref.astype(np.float32))}")
    # every position distinct: A[i,k] = i + 1 at k = 0 only, B = identity-like rows -> catches any permutation
    a = np.zeros((16, 4), np.float16); a[:, 0] = np.arange(1, 17); b = np.zeros((4, 16), np.float16); b[0] = np.arange(16) * 32
    c = np.arange(256, dtype=np.float32).reshape(16, 16) * 0.5
    d, ref = check(a, b, c)
    print(f"permutation probe: exact={np.array_equal(d, ref.astype(np.float32))}")
    a = rng.standard_normal((16, 4)).astype(np.float16); b = rng.standard_normal((4, 16)).astype(np.float16)
    c = rng.standard_normal((16, 16)).astype(np.float32)
    d, ref = check(a, b, c)
    print(f"normal: max|D - fp64 oracle| = {np.abs(d - ref).max():.3e}")
