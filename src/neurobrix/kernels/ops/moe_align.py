"""Device-side MoE token alignment — constant-tuple, deterministic.

Replaces the host moe_align (D2H of topk_ids -> numpy bincount/scatter
-> H2D of tables) with three tiny Triton kernels so the whole routing
pipeline stays on device. Two independent wins:

1. Replay eligibility (P-REPLAY-KV-DECODE): the host align was a D2H
   read of router output per MoE layer per step — a structural
   plan-breaker (recorded H2D tables would replay ONE step's expert
   choice forever; the loud refusal at recording caught exactly this).
   Device-side, the recorded launches recompute the tables from the
   CURRENT router output at every replay.
2. Sync elimination: 48 D2H round-trips per decode step disappear from
   the MoE band (same class as the device-scalar workstream).

Pattern source: vLLM's moe_align_block_size CUDA kernel (grid sized on
the worst case, consumers early-exit on the device num_tokens_post_
padded scalar — that guard already exists in all three fused variants).
vLLM's atomic rank assignment is NOT reproduced: atomics permute tokens
within an expert run-to-run. Here the rank is a deterministic pairwise
count (#earlier tokens with the same expert), which reproduces the host
sort's order EXACTLY — the byte gate against the removed host path
adjudicates the swap directly. n = M * top_k is tiny (8 at decode), so
the O(n^2) rank costs nothing at the scale it runs.

Outputs are sized on the shape-derived worst case
min(n * BS, n + E * (BS - 1)) rounded up to BS — the true total lives
ONLY in the device scalar. All launch tuples are constant per shape
class by construction.
"""

import triton
import triton.language as tl


@triton.jit
def moe_align_stage1_kernel(
    topk_ids_ptr,        # [n] int64 — flat expert index per (token, k)
    offsets_ptr,         # [E] int64 out — exclusive padded starts
    padded_ptr,          # [E] int64 out — per-expert padded count
    num_post_pad_ptr,    # [1] int64 out — total padded entries
    n,
    BS: tl.constexpr,    # block size (fused kernel BLOCK_SIZE_M)
    BE: tl.constexpr,    # padded expert count (pow2 >= E)
    E: tl.constexpr,     # true expert count
    BT: tl.constexpr,    # token chunk width
    BF: tl.constexpr,    # expert chunk width (divides BE)
):
    """Single program: per-expert counts -> padded counts -> exclusive
    prefix offsets (masked triangular sum — no tl.cumsum dependency).
    Worked over expert chunks of BF: each chunk's padded counts go to the
    prefix sum as a [BE, BF] block, so no [BE, BE] or [BE, BT] matrix is
    held; the host sizes BT and BF to the backend's threadgroup memory
    (Metal 32 KB: 128 experts at BT = BF = 128 asked 64 KB, Qwen3-30B-A3B)."""
    e = tl.arange(0, BE)
    excl = tl.zeros((BE,), dtype=tl.int32)
    total = tl.sum(tl.zeros((BF,), dtype=tl.int32), axis=0)
    for f0 in tl.static_range(0, BE, BF):
        f = f0 + tl.arange(0, BF)
        counts = tl.zeros((BF,), dtype=tl.int32)
        for start in range(0, n, BT):
            t = start + tl.arange(0, BT)
            ids = tl.load(topk_ids_ptr + t, mask=t < n, other=-1)
            m = (ids[None, :] == f[:, None]) & (t[None, :] < n)
            counts += tl.sum(m.to(tl.int32), axis=1)
        padded = ((counts + BS - 1) // BS) * BS
        mask_f = f < E
        tl.store(padded_ptr + f, padded.to(tl.int64), mask=mask_f)
        total += tl.sum(tl.where(mask_f, padded, 0), axis=0)
        tri = f[None, :] < e[:, None]
        excl += tl.sum(tl.where(tri, padded[None, :], 0), axis=1)
    tl.store(offsets_ptr + e, excl.to(tl.int64), mask=e < E)
    tl.store(num_post_pad_ptr, total.to(tl.int64))


@triton.jit
def moe_align_stage2_kernel(
    offsets_ptr,         # [E] int64
    padded_ptr,          # [E] int64
    sorted_ids_ptr,      # [max_total] int64 out — sentinel-filled here
    expert_ids_ptr,      # [max_blocks] int64 out — expert per block, -1 pad
    n,
    max_total,
    BS: tl.constexpr,
    BE: tl.constexpr,
    E: tl.constexpr,
    BLK: tl.constexpr,   # positions per program (multiple of BS)
    BF: tl.constexpr,    # expert chunk width (divides BE)
):
    """Grid over output positions: sentinel-fill sorted ids and resolve
    each block's owning expert (offsets[e] <= block_start < offsets[e]
    + padded[e]); blocks past the true total get -1 and are skipped by
    the fused kernels' num_tokens_post_padded early-exit anyway. The
    experts are read in chunks of BF ([BLK, BF] blocks, sized by the host
    to the backend's threadgroup memory, as stage 1)."""
    pid = tl.program_id(0).to(tl.int64)
    p = pid * BLK + tl.arange(0, BLK)
    in_p = p < max_total
    tl.store(sorted_ids_ptr + p, tl.zeros((BLK,), dtype=tl.int64) + n,
             mask=in_p)
    b = p // BS
    pb = (b * BS).to(tl.int64)
    eid = tl.zeros((BLK,), dtype=tl.int32)
    hits = tl.zeros((BLK,), dtype=tl.int32)
    for f0 in tl.static_range(0, BE, BF):
        e = f0 + tl.arange(0, BF)
        mask_e = e < E
        offs = tl.load(offsets_ptr + e, mask=mask_e, other=0)
        padd = tl.load(padded_ptr + e, mask=mask_e, other=0)
        m = (pb[:, None] >= offs[None, :]) \
            & (pb[:, None] < (offs + padd)[None, :]) & mask_e[None, :]
        eid += tl.sum(tl.where(m, e[None, :], 0), axis=1)
        hits += tl.sum(m.to(tl.int32), axis=1)
    eid = tl.where(hits > 0, eid, -1)
    write_b = in_p & (p % BS == 0)
    tl.store(expert_ids_ptr + b, eid.to(tl.int64), mask=write_b)


@triton.jit
def moe_align_stage3_kernel(
    topk_ids_ptr,        # [n] int64
    offsets_ptr,         # [E] int64
    sorted_ids_ptr,      # [max_total] int64 — scatter over the sentinels
    n,
    E: tl.constexpr,     # true expert count (out-of-range ids skipped,
                         # mirroring the host scatter's bounds check)
    BLKT: tl.constexpr,  # tokens per program
    BN: tl.constexpr,    # comparison chunk width
):
    """Grid over tokens: deterministic rank (#earlier tokens with the
    same expert) -> scatter token index to offsets[expert] + rank.
    Order-preserving, no atomics — byte-identical to the host sort it
    replaces. MUST launch after stage 2 (stream-serialized): it
    overwrites sentinel positions."""
    pid = tl.program_id(0).to(tl.int64)
    i = pid * BLKT + tl.arange(0, BLKT)
    in_i = i < n
    my = tl.load(topk_ids_ptr + i, mask=in_i, other=-1)
    in_i = in_i & (my >= 0) & (my < E)
    rank = tl.zeros((BLKT,), dtype=tl.int32)
    for start in range(0, n, BN):
        j = start + tl.arange(0, BN)
        idj = tl.load(topk_ids_ptr + j, mask=j < n, other=-2)
        m = (idj[None, :] == my[:, None]) & (j[None, :] < i[:, None])
        rank += tl.sum(m.to(tl.int32), axis=1)
    off_e = tl.load(offsets_ptr + my, mask=in_i, other=0)
    pos = off_e + rank
    tl.store(sorted_ids_ptr + pos, i.to(tl.int64), mask=in_i)
