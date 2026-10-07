# Tensor cores through linear layouts — the m8n8k4 matrix unit on sm_70

**Why.** Triton >= 3.3 lowers `tl.dot` to scalar FMA below sm_80 (triton-lang/triton#5066). On a V100 every
matrix product of the Triton mode therefore ran on the CUDA cores: the engine's flash attention measured 2.46
TFLOP/s at T = 79 200, D = 96, about 94 % of an Allegro denoise step. The V100's tensor cores (HMMA.884) are
reached instead from our own Gluon kernels, with no native code outside Triton, no external library, no cubin
loaded on the side and no Triton downgrade: Gluon (`triton.experimental.gluon`, Triton 3.8) has no sm_70 gate,
and ptxas accepts `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32` for sm_70.

**Selection.** The hardware profile declares the unit; no code names a card:

```yaml
# src/neurobrix/config/vendors/nvidia/volta.yml
matrix_unit:
  shape: m8n8k4
  operand_dtype: float16
  accumulator_dtype: float32
  flash:
    - head_dim_le: 256
      block_m: 64
      block_n: 32
```

`kernels/ops/_configs.py` reads it (`matrix_unit()`, `matrix_unit_flash_tile(head_dim)`); the SDPA wrapper takes
the m8n8k4 flash when the profile declares the shape, a tile row covers the head dim and the operands are in the
declared dtype. A profile without `matrix_unit` keeps the `tl.dot` kernels unchanged.

## The fragment

From CUTLASS `cute/atom/mma_traits_sm70.hpp`, `SM70_8x8x4_F32F16F16F32_TN`. A lane is
`a0 + 2 a1 + 4 q0 + 8 q1 + 16 b`; the four quadpairs `q` each compute one 8x8 quadrant of a 16x16 tile. Thread
`t = a0 + 2 a1 + 4 b` of a quadpair holds row `t` of A and column `t` of B for `k = 0..3`; its accumulator register
`v = v0 + 2 v1 + 4 v2` holds `D[m = a0 + 2 v1 + 4 b, n = v0 + 2 a1 + 4 v2]`.

## The accumulator as a 7-D tensor

A `[BM, BN]` accumulator is held as `[MB, 2, 8, NB, 2, 2, 4]` = `(mb, mi3, mi2..0, nb, ni3, ni2, ni1..0)` with
`m = 16 mb + mi`, `n = 16 nb + ni`. Its row-major order is `[BM, BN]`'s, so a reshape moves nothing. In these
coordinates the A fragment depends only on `(mb, mi3, mi2..0, ni1..0)` and the B fragment only on
`(mi2..0, nb, ni3, ni1..0)`: each operand is a plain broadcast of `[MB, 2, 8, 1, 1, 1, 4]` and
`[1, 1, 8, NB, 2, 1, 4]`, which `inline_asm_elementwise` broadcasts itself (`pack = 8`, eight accumulator elements
per call). One fragment register therefore feeds every sub-tile that shares it, with no shuffle.

The accumulator layout is a `DistributedLinearLayout` written from the fragment (`mma_m8n8k4.acc_layout`): register
bases `ni0, mi1, ni2` then the NB and MB repeats, lane bases `mi0, ni1, mi3, ni3, mi2`, warp bases over MB then NB.

## Operands through shared memory

Global tiles are loaded coalesced and staged in shared memory as `[k-step x rows, 4]` (k innermost, 8 bytes per
row, read back as LDS.64). One k-step is a `.slice(row0, rows)` of a 2-D buffer and is loaded in a linear layout
derived from the accumulator's (`frag_a_layout`, `frag_b_layout`), then reshaped and permuted in registers to the
broadcast shape; `convert_layout(assert_trivial=True)` proves that this relabelling moves no data. Constraints met
on the way: shared buffers are 2-D (a rank-3 `.index()` keeps the rank-3 order), every dimension is a power of two,
and a global constant is wrapped in `gl.constexpr`.

**Bank conflicts.** An operand whose k runs along rows (B of a matmul, V of attention) is loaded with
`BlockedLayout([4, 2], [1, 32], [nw, 1], [1, 0])`, so each thread holds four consecutive k of a column and the
transposing store is one STS.64. Without it the store was 16-way conflicted: 39 -> 58 TFLOP/s.

## Attention on the block

`kernels/ops/flash_attention_m8n8k4.py` keeps the contract of `flash_attention.py` (Q scaled by `softmax_scale` and
rounded to its own dtype, memory-resident additive bias, GQA groups, log-sum-exp stored, tails masked, a fully
masked row left to the wrapper's guard). One warp owns 16 query rows, so the row statistics never cross a warp;
`S = Q K^T` and `O += P V` are HMMA; the online softmax runs in the log2 domain on the accumulator layout (row
statistics are its four column dims reduced). The head dim is split into at most two powers of two (96 = 64 + 32),
never padded to the next power.

## Launching

The engine's launcher binds kernels without `JITFunction.create_binder`, which is where Triton chooses
`GluonASTSource` for a `@gluon.jit` kernel; the launcher makes the same choice (`launcher._source`), otherwise the
module lacks `ttg.num-warps` and layout verification fails.

## Proofs and numbers (one V100-SXM2, 1290 MHz)

HMMA ceiling at that clock: 80 SM x 8 tensor cores x 128 FLOP/clk x 1.290 GHz = 105.7 TFLOP/s (the 125 figure is
the 1530 MHz boost).

| step | proof | number |
|---|---|---|
| 1. one 16x16x4 tile | integers bit-exact; SASS 4 HMMA.884; a wrong register basis injected -> error 148, restored -> exact | — |
| 2. matmul block | integers bit-exact, normal rel 1.7e-6 vs fp64 | 4096^3, 128x128x32, 4 warps: 57.9 TFLOP/s (cuBLAS 84.9) |
| 3. flash forward | vs fp64 at T777 D64/80/128 and T1000 D96: 2.8e-5..4.0e-5; vs the FMA flash at T79200 D96 H4: 8.1e-6 | 43.6 TFLOP/s vs 2.46 (x17.7) |

The probes that produced them are in `tools/tensor_cores_m8n8k4/` with their scan outputs.
