"""Vector (SIMT) MoE decode band — M=1, int4 g128 asym, no tl.dot.

Fourth member of the SIMT decode family (decode_attn_vec + gemv_vec,
both adopted). The decode MoE band ran as three Grouped-GEMM launches
over sorted-token tables: at M=1 with 8 active experts of 128, every
16-row MMA tile carries ONE real row (15/16 M-waste), plus the 3-stage
moe_align table build, a separate SwiGLU pass, and a sum_wrapper
combine — 18.4 ms of a 38.4 ms step for ~944 MB of weight reads
(~12x the streaming floor; full_step_attribution_2026_08_24).

The 2026-08-24 web research (R16) confirms the regime: vLLM itself
abandoned its grouped Triton kernel at small m (its CUDA moe_wna16 —
SIMT half2 FMA, one thread per output column, group-boundary scale
loads — wins +41% at bs=1 on A100/DeepSeek; our num_seqs*top_k/E =
0.0625 is two orders below its own switchover). Its combine uses
atomicAdd (order-nondeterministic) — REFUSED here; the deterministic
production forms fuse only the routing-weight MULTIPLY into the
down epilogue (vLLM Triton MUL_ROUTED_WEIGHT) and keep the sum in
fixed order. This kernel pair goes one step further and keeps the
cross-expert sum INSIDE the down kernel as a FIXED-ORDER expert loop —
no intermediate per-slot outputs, no sum_wrapper, no atomics.

Two kernels replace the whole band:
  1. `moe_gateup_vec_kernel` — grid (TOP_K, cdiv(INTER, BLOCK_N)):
     each program owns one active expert's tile of the intermediate
     dim, runs TWO dequant-matvec accumulators (gate and up — the
     llama.cpp dual-accumulator GLU shape, no weight-layout change)
     over K with int4-g128-asym dequant in the FMA loop, and applies
     SwiGLU in the epilogue: h[e, tile] = silu(g) * u.
  2. `moe_down_combine_vec_kernel` — grid (cdiv(HID, BLOCK_N),):
     each program accumulates over the TOP_K experts IN FIXED ORDER:
     acc += w_e * (dequant(W_down_e)[:, tile]^T @ h[e]) — the router
     weight fused as the epilogue multiply, the sum deterministic by
     construction.
  moe_align disappears at M=1 (no token sort exists to build); the
  SwiGLU pass, the sum_wrapper combine, and the zero-init of the
  grouped caches disappear with it.

Dequant expression TEXTUALLY IDENTICAL to dequant_gemv_int4_kernel
(the parity-derived brick: q unpacked LSB-nibble from int32,
w = q*scale + qmin in pure fp32, masked tails contribute exact +0.0).
Expert weight/scale/qmin pointers come from the existing PtrTables
int64 tables (tl.cast(..., tl.pointer_type, bitcast=True) — the
established idiom).
"""

import triton
import triton.language as tl


@triton.jit
def moe_gateup_vec_kernel(
    x_ptr,            # [K] activation (fp16/fp32)
    topk_ids_ptr,     # [TOP_K] int — active expert ids, fixed order
    gate_qw_tab,      # [E] int64 — qweight ptrs   (int32 [K//8, N])
    gate_sc_tab,      # [E] int64 — scales ptrs    (fp16 [G, N])
    gate_mn_tab,      # [E] int64 — qmins ptrs     (fp16 [G, N])
    up_qw_tab, up_sc_tab, up_mn_tab,
    h_ptr,            # [TOP_K, N] fp32 out — silu(gate) * up
    K, N,
    stride_wk, stride_wn,     # packed-int32 strides (shared gate/up)
    stride_sg, stride_sn,     # scales/qmins strides
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,    # divides GROUP (128)
    GROUP: tl.constexpr,
    PACK: tl.constexpr,       # 8 nibbles per int32
):
    pid_e = tl.program_id(0)
    pid_n = tl.program_id(1)
    eid = tl.load(topk_ids_ptr + pid_e).to(tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    g_qw = tl.cast(tl.load(gate_qw_tab + eid),
                   tl.pointer_type(tl.int32), bitcast=True)
    g_sc = tl.cast(tl.load(gate_sc_tab + eid),
                   tl.pointer_type(tl.float16), bitcast=True)
    g_mn = tl.cast(tl.load(gate_mn_tab + eid),
                   tl.pointer_type(tl.float16), bitcast=True)
    u_qw = tl.cast(tl.load(up_qw_tab + eid),
                   tl.pointer_type(tl.int32), bitcast=True)
    u_sc = tl.cast(tl.load(up_sc_tab + eid),
                   tl.pointer_type(tl.float16), bitcast=True)
    u_mn = tl.cast(tl.load(up_mn_tab + eid),
                   tl.pointer_type(tl.float16), bitcast=True)

    acc_g = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for kc in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = kc * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        mask = mask_k[:, None] & mask_n[None, :]
        a = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        g = kc * BLOCK_K // GROUP  # BLOCK_K divides GROUP
        w_off = ((offs_k[:, None] // PACK) * stride_wk
                 + offs_n[None, :] * stride_wn)
        shift = (offs_k[:, None] % PACK) * 4

        packed = tl.load(g_qw + w_off, mask=mask, other=0)
        q = (packed >> shift) & 0xF
        scale = tl.load(g_sc + g * stride_sg + offs_n * stride_sn,
                        mask=mask_n, other=0.0)
        mn = tl.load(g_mn + g * stride_sg + offs_n * stride_sn,
                     mask=mask_n, other=0.0)
        w = (q.to(tl.float32) * scale[None, :].to(tl.float32)
             + mn[None, :].to(tl.float32))
        b = tl.where(mask, w, 0.0)
        acc_g += tl.sum(a[:, None] * b, axis=0)

        packed = tl.load(u_qw + w_off, mask=mask, other=0)
        q = (packed >> shift) & 0xF
        scale = tl.load(u_sc + g * stride_sg + offs_n * stride_sn,
                        mask=mask_n, other=0.0)
        mn = tl.load(u_mn + g * stride_sg + offs_n * stride_sn,
                     mask=mask_n, other=0.0)
        w = (q.to(tl.float32) * scale[None, :].to(tl.float32)
             + mn[None, :].to(tl.float32))
        b = tl.where(mask, w, 0.0)
        acc_u += tl.sum(a[:, None] * b, axis=0)

    # SwiGLU epilogue: silu(gate) * up, pure fp32.
    h = acc_g * tl.sigmoid(acc_g) * acc_u
    tl.store(h_ptr + pid_e * N + offs_n, h, mask=mask_n)


@triton.jit
def moe_down_combine_vec_kernel(
    h_ptr,            # [TOP_K, Kd] fp32 — kernel 1's output
    topk_ids_ptr,     # [TOP_K] int
    topk_w_ptr,       # [TOP_K] routing weights (fp16/fp32)
    down_qw_tab, down_sc_tab, down_mn_tab,
    out_ptr,          # [N] out (fp32)
    Kd, N,
    stride_wk, stride_wn,
    stride_sg, stride_sn,
    TOP_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP: tl.constexpr,
    PACK: tl.constexpr,
):
    """The combine IS the kernel: experts iterated in FIXED order, the
    routing weight applied as the per-expert epilogue multiply — the
    cross-expert sum is deterministic by construction, and no
    intermediate per-slot output or separate reduction exists."""
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for e in tl.static_range(TOP_K):
        eid = tl.load(topk_ids_ptr + e).to(tl.int64)
        we = tl.load(topk_w_ptr + e).to(tl.float32)
        qw = tl.cast(tl.load(down_qw_tab + eid),
                     tl.pointer_type(tl.int32), bitcast=True)
        sc = tl.cast(tl.load(down_sc_tab + eid),
                     tl.pointer_type(tl.float16), bitcast=True)
        mp = tl.cast(tl.load(down_mn_tab + eid),
                     tl.pointer_type(tl.float16), bitcast=True)
        part = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for kc in range(0, tl.cdiv(Kd, BLOCK_K)):
            offs_k = kc * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_k = offs_k < Kd
            mask = mask_k[:, None] & mask_n[None, :]
            a = tl.load(h_ptr + e * Kd + offs_k, mask=mask_k,
                        other=0.0)
            g = kc * BLOCK_K // GROUP
            packed = tl.load(qw + (offs_k[:, None] // PACK) * stride_wk
                             + offs_n[None, :] * stride_wn,
                             mask=mask, other=0)
            q = (packed >> ((offs_k[:, None] % PACK) * 4)) & 0xF
            scale = tl.load(sc + g * stride_sg + offs_n * stride_sn,
                            mask=mask_n, other=0.0)
            mn = tl.load(mp + g * stride_sg + offs_n * stride_sn,
                         mask=mask_n, other=0.0)
            w = (q.to(tl.float32) * scale[None, :].to(tl.float32)
                 + mn[None, :].to(tl.float32))
            b = tl.where(mask, w, 0.0)
            part += tl.sum(a[:, None] * b, axis=0)
        acc += we * part

    tl.store(out_ptr + offs_n, acc, mask=mask_n)
