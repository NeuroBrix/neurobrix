"""E7 — fused int4 dequant-GEMV family (quantized tier, pure Triton).

Weight-only int4, groupwise g=128, ASYMMETRIC: per group along K an
fp16 scale and an fp16 min; dequant is `q * scale + min` with q an
UNSIGNED nibble (llama.cpp q4_1's exact form — no -8 offset, no
AutoGPTQ +1 storage quirk). Packing: 8 nibbles per int32, flat
low-nibble-first (nibble i of word r encodes element k = 8*r + i),
W_q laid out (K//8, N) K-major so loads coalesce along N — the
GemLite convention.

THE TIER'S BYTE ORACLE (scoping "quantized tier contract", clause 1):
the fused kernel must be byte-identical to dequantize-then-GEMV. A
cuBLAS/torch reference can never be byte-matched (its reduction order
is internal); the oracle is therefore built from THIS file's own
kernels: `dequant_int4_kernel` (standalone dequant whose dequant
expression is textually identical to the fused kernel's) feeding
`gemv_ref_kernel` (same BLOCK_N/BLOCK_K, same ascending-K chunk
loop, same tl.sum tree, same fp32 accumulator). Fused == oracle by
construction; the microtest proves it byte-for-byte with ONE pinned
config for both kernels (autotune would change the tl.sum tree — E7
GEMV sits OUTSIDE the mm/bmm/addmm/conv2d autotune sanction).

sm_70 discipline (sourced, R38 study + 2026-08-16 research): M=1
decode GEMV, no tl.dot (Volta fp16 dot lowers to FMA anyway), SPLIT_K
= 1 — a single program per N-tile walks all of K, NO atomics
(atomic-splitK is arrival-order nondeterministic and fp16 atomics are
version-fragile on our pin; both break the byte contract). The win is
weight bandwidth: 4x fewer weight bytes than fp16. num_stages <= 2
(cp.async pipelining is sm_80+; more stages only add register
pressure to the unpack-heavy loop).

Constraints enforced by the wrappers: K % BLOCK_K == 0 handled by
masking; BLOCK_K in {32, 64, 128} divides group_size=128 so a K-chunk
never crosses a group boundary (one scale/min scalar per (chunk, n) —
the GemLite pruner rule).
"""

import triton
import triton.language as tl

GROUP_SIZE: int = 128
PACK: int = 8  # nibbles per int32

# ONE pinned config for the byte-parity path (oracle AND fused run
# with the same values — the tl.sum reduction tree must be identical).
# Perf sweeps may explore {32,64,128} x {32,64,128} x warps {1,2,4}
# in a measurement harness, never in the parity gate.
BLOCK_N: int = 64
BLOCK_K: int = 128
NUM_WARPS: int = 4


@triton.jit
def dequant_int4_kernel(
    wq_ptr, scales_ptr, mins_ptr, out_ptr,
    K, N,
    stride_wk, stride_wn,
    stride_sg, stride_sn,
    stride_ok, stride_on,
    BLOCK_K_C: tl.constexpr, BLOCK_N_C: tl.constexpr,
    GROUP_C: tl.constexpr, PACK_C: tl.constexpr,
):
    """Standalone dequant: packed int4 -> dense fp32 [K, N].

    Oracle part 1. The dequant expression below is the TEXTUAL
    contract shared with the fused kernel: same loads, same shifts,
    same pure-fp32 dtype path.
    """
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_k = pid_k * BLOCK_K_C + tl.arange(0, BLOCK_K_C)
    offs_n = pid_n * BLOCK_N_C + tl.arange(0, BLOCK_N_C)
    mask_k = offs_k < K
    mask_n = offs_n < N
    mask = mask_k[:, None] & mask_n[None, :]

    packed = tl.load(
        wq_ptr + (offs_k[:, None] // PACK_C) * stride_wk
        + offs_n[None, :] * stride_wn, mask=mask, other=0)
    q = (packed >> ((offs_k[:, None] % PACK_C) * 4)) & 0xF

    g = pid_k * BLOCK_K_C // GROUP_C  # BLOCK_K_C divides GROUP_C
    scale = tl.load(scales_ptr + g * stride_sg + offs_n * stride_sn,
                    mask=mask_n, other=0.0)
    mn = tl.load(mins_ptr + g * stride_sg + offs_n * stride_sn,
                 mask=mask_n, other=0.0)
    # Canonical dequant: PURE fp32 (llama.cpp dmmv default,
    # dfloat=float — the fp16 weight never exists at runtime).
    # Contraction-immune: q*scale is EXACT in fp32 (4-bit int x fp16
    # mantissa), so fma(q,s,m) == q*s+m bit-for-bit whatever the
    # compiler emits. The dense oracle tensor is fp32.
    w = (q.to(tl.float32) * scale[None, :].to(tl.float32)
         + mn[None, :].to(tl.float32))

    tl.store(out_ptr + offs_k[:, None] * stride_ok
             + offs_n[None, :] * stride_on, w, mask=mask)


@triton.jit
def gemv_ref_kernel(
    x_ptr, w_ptr, out_ptr,
    K, N,
    stride_wk, stride_wn,
    BLOCK_K_C: tl.constexpr, BLOCK_N_C: tl.constexpr,
):
    """Reference dense-weight GEMV: out[n] = sum_k x[k] * w[k, n].

    Oracle part 2. Fixed reduction structure — ascending-K chunk loop,
    per-chunk tl.sum over axis 0, fp32 accumulator, one store — the
    EXACT structure of the fused kernel below. (A nibble-major
    restructure with per-nibble partial sums was MEASURED and
    reverted: it slowed the fused kernel 0.090 -> 0.224 ms at
    4096x11008 — the lane-selection sums cost more than the redundant
    packed loads they save; the fused kernel is ALU-issue-bound, not
    load-bound.)
    """
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N_C + tl.arange(0, BLOCK_N_C)
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_N_C,), dtype=tl.float32)
    for kc in range(0, tl.cdiv(K, BLOCK_K_C)):
        offs_k = kc * BLOCK_K_C + tl.arange(0, BLOCK_K_C)
        mask_k = offs_k < K
        a = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0)
        b = tl.load(w_ptr + offs_k[:, None] * stride_wk
                    + offs_n[None, :] * stride_wn,
                    mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.sum(a.to(tl.float32)[:, None] * b.to(tl.float32),
                      axis=0)
    tl.store(out_ptr + offs_n, acc, mask=mask_n)


@triton.jit
def dequant_gemv_int4_kernel(
    x_ptr, wq_ptr, scales_ptr, mins_ptr, out_ptr,
    K, N,
    stride_wk, stride_wn,
    stride_sg, stride_sn,
    BLOCK_K_C: tl.constexpr, BLOCK_N_C: tl.constexpr,
    GROUP_C: tl.constexpr, PACK_C: tl.constexpr,
):
    """Fused: out[n] = sum_k x[k] * dequant(wq)[k, n], fp32 accum.

    Dequant expression textually identical to dequant_int4_kernel;
    reduction structure textually identical to gemv_ref_kernel —
    byte-equal to the two-step oracle by construction (proven by the
    parity microtest with the pinned config). The packed load gathers
    `offs_k // 8` (each int32 requested 8x, absorbed by L1) — the
    nibble-major single-load variant was MEASURED SLOWER (see the
    oracle kernel's docstring); the kernel is ALU-issue-bound.
    """
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N_C + tl.arange(0, BLOCK_N_C)
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_N_C,), dtype=tl.float32)
    for kc in range(0, tl.cdiv(K, BLOCK_K_C)):
        offs_k = kc * BLOCK_K_C + tl.arange(0, BLOCK_K_C)
        mask_k = offs_k < K
        mask = mask_k[:, None] & mask_n[None, :]
        a = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0)

        packed = tl.load(
            wq_ptr + (offs_k[:, None] // PACK_C) * stride_wk
            + offs_n[None, :] * stride_wn, mask=mask, other=0)
        q = (packed >> ((offs_k[:, None] % PACK_C) * 4)) & 0xF
        g = kc * BLOCK_K_C // GROUP_C  # BLOCK_K_C divides GROUP_C
        scale = tl.load(scales_ptr + g * stride_sg + offs_n * stride_sn,
                        mask=mask_n, other=0.0)
        mn = tl.load(mins_ptr + g * stride_sg + offs_n * stride_sn,
                     mask=mask_n, other=0.0)
        # Canonical dequant — textually identical to
        # dequant_int4_kernel: PURE fp32, no fp16 materialization
        # (contraction-immune: q*scale exact in fp32).
        w = (q.to(tl.float32) * scale[None, :].to(tl.float32)
             + mn[None, :].to(tl.float32))
        # Masked K-tail rows dequant to `mn` (q=0), not 0 — zero them
        # through the mask so the tail contributes exact +0.0, same as
        # the oracle GEMV's masked b-load.
        b = tl.where(mask, w, 0.0)
        acc += tl.sum(a.to(tl.float32)[:, None] * b, axis=0)
    tl.store(out_ptr + offs_n, acc, mask=mask_n)
