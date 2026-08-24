"""Vector (SIMT) fp16 GEMV — the decode projection kernel, no tl.dot.

Second member of the SIMT decode family (after ops/decode_attn_vec.py,
adopted 2026-08-23). The FlagGems-ported `mv_kernel` runs the row's
193 per-token projections at ~280 GB/s effective (7.1 ms/token, 36 us
per call measured as IN-GRAPH kernel time — launch overhead already
excluded by the CUDA-graph replay, so the cost is kernel structure):
its (BLOCK_N=64, BLOCK_M=256) fp32 accumulator tile is 64 KB of
registers per CTA (spill class), and its N/64 grid puts 32 programs on
80 SMs at N=2048.

The 2026-08-24 web research (R16) says the structure for OUR layout —
weights pre-transposed to row-major [N, K], K contiguous per output
row — is llama.cpp's `mmvf.cu` shape: a program owns a small group of
output rows, its threads stride K with wide vectorized loads, fp32
scalar accumulators, fixed-order reduction (llama.cpp: warp shuffle +
smem, deterministic, no atomics; grid = rows). GemLite's Triton GEMV
splits K across programs and reduces with `tl.atomic_add` — refused
here: run-to-run fp ordering nondeterminism, the exact class the
replay verify and the drift gate forbid. TRT-LLM's cudaCoreGemm keeps
all of K inside one block for the same reason.

Triton translation: acc is (BLOCK_N,) — reduced PER K-CHUNK
(`tl.sum(a_tile * b_chunk[None, :], 1)`), 256x less register state
than the (BLOCK_N, BLOCK_K) end-reduced tile; BLOCK_N small (8-16
rows) so the N-grid alone fills the 80 SMs (N=2048 -> 128-256
programs); `tl.multiple_of`/`tl.max_contiguous` hints on the
K-contiguous weight offsets so ptxas emits 128-bit loads (llama.cpp
PR #9816: vectorizing exactly these loads was worth 1.27x on HBM
parts); num_stages=1 (no async-copy engine on sm_70).

Deterministic by construction: one program per output-row group, all
of K reduced inside it in fixed chunk order, no atomics.
"""

import triton
import triton.language as tl


@triton.jit
def gemv_vec_kernel(
    A_ptr,          # [N, K] weights, row-major (pre-transposed), fp16
    B_ptr,          # [K] activation (fp32 on Volta via the AMP upcast)
    C_ptr,          # [N] out
    N, K,
    stride_an, stride_am,
    stride_bm,
    stride_cn,
    BLOCK_N: tl.constexpr,   # output rows per program (small: 8-16)
    BLOCK_K: tl.constexpr,   # K-chunk width (wide: 128-512)
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_K)
    # The weight rows are K-contiguous (stride_am == 1 on the
    # pre-transposed layout): tell the compiler so the loads vectorize
    # to 128-bit transactions.
    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

    a_ptrs = A_ptr + offs_n[:, None] * stride_an + offs_k[None, :] * stride_am
    b_ptrs = B_ptr + offs_k * stride_bm
    for k0 in range(0, K, BLOCK_K):
        mask_k = k0 + offs_k < K
        a = tl.load(a_ptrs, mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        acc += tl.sum(a * b[None, :], 1)
        a_ptrs += BLOCK_K * stride_am
        b_ptrs += BLOCK_K * stride_bm

    tl.store(C_ptr + offs_n * stride_cn, acc, mask=mask_n)
