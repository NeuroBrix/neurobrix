"""The expert store's residency kept ON THE CARD for a decode token (S3, 2026-10-28).

An expert store serves the routed layers no card holds whole from a cache of device slots. Its host-side residency
(`core/runtime/expert_residency.ExpertResidency`) reads every call's routing back to the host, decides the misses, and
copies them — a host round trip per layer per token that also keeps the decode step from being replayed (Qwen3-30B-A3B
on 4 V100: 8 store layers ~32 ms of a ~80 ms token). For one token the same LRU runs here:

  1. `expert_slot_assign_kernel` — ONE program, deterministic: every routed expert already resident is a hit (its stamp
     refreshed to the clock); every miss takes the free slot or the least recently stamped slot this call does not read
     (a slot this call reads carries the current clock, so it is never the oldest), evicting its previous owner from
     the slot map. Writes each routed expert's slot, and for a miss the expert to copy (-1 for a hit). The clock moves.
  2. `expert_miss_copy_kernel` — grid (TOP_K, 3 projections, chunks): a miss copies its expert's three projections from
     page-locked host memory (read through its device-visible address) into its slot; a hit returns at once.

The decode band then reads the experts through the slots' fixed pointer table, indexed by slot (`moe_decode_vec` half
kernels). Every launch is the same at every token — pointers to persistent tables — so the step replays.
"""

import triton
import triton.language as tl


@triton.jit
def expert_slot_assign_kernel(
    ids_ptr,          # [TOP_K] int64 — the routed experts of this token, the router's order
    slot_of_ptr,      # [L * E] int32 — slot holding (layer, expert), -1 when none
    owner_ptr,        # [S] int32 — the (layer * E + expert) a slot holds, -1 when free
    stamp_ptr,        # [S] int64 — the clock at a slot's last use, -1 when free
    clock_ptr,        # [1] int64
    slot_out_ptr,     # [TOP_K] int64 — the slot of each routed expert
    miss_out_ptr,     # [TOP_K] int64 — the expert to copy into that slot, -1 for a hit
    layer_base,       # layer index * E
    S,
    TOP_K: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    clock = tl.load(clock_ptr)
    offs = tl.arange(0, BLOCK_S)
    in_range = offs < S
    # hits first: their slots carry the current clock before any miss looks for a victim
    for k in tl.static_range(TOP_K):
        e = tl.load(ids_ptr + k)
        s = tl.load(slot_of_ptr + layer_base + e)
        if s >= 0:
            tl.store(stamp_ptr + s, clock)
            tl.store(slot_out_ptr + k, s.to(tl.int64))
            tl.store(miss_out_ptr + k, -1)
    tl.debug_barrier()
    for k in tl.static_range(TOP_K):
        e = tl.load(ids_ptr + k)
        s = tl.load(slot_of_ptr + layer_base + e)
        if s < 0:
            stamps = tl.load(stamp_ptr + offs, mask=in_range, other=9223372036854775807)
            victim = tl.argmin(stamps, axis=0)
            old = tl.load(owner_ptr + victim)
            if old >= 0:
                tl.store(slot_of_ptr + old, -1)
            tl.store(owner_ptr + victim, (layer_base + e).to(tl.int32))
            tl.store(slot_of_ptr + layer_base + e, victim.to(tl.int32))
            tl.store(stamp_ptr + victim, clock)
            tl.store(slot_out_ptr + k, victim.to(tl.int64))
            tl.store(miss_out_ptr + k, e)
        tl.debug_barrier()
    tl.store(clock_ptr, clock + 1)


@triton.jit
def expert_miss_copy_kernel(
    miss_ptr,         # [TOP_K] int64 — expert to copy, -1 for a hit
    slot_ptr,         # [TOP_K] int64 — its slot
    host_g, host_u, host_d,   # [L * E] int64 — device-visible addresses of the host experts' projections
    slot_g, slot_u, slot_d,   # [S] int64 — the slots' projection addresses
    layer_base,
    n_g, n_u, n_d,    # elements per projection
    BLOCK: tl.constexpr,
    W_BF16: tl.constexpr,
):
    k = tl.program_id(0).to(tl.int64)
    p = tl.program_id(1).to(tl.int64)
    c = tl.program_id(2).to(tl.int64)
    e_raw = tl.load(miss_ptr + k)
    # a hit (-1) copies nothing: masked, not returned (a bare return is unstructured control flow some backends do
    # not lower); its address loads read row 0 of the tables, a valid index, and the copy's mask is all false
    miss = e_raw >= 0
    e = tl.where(miss, e_raw, 0)
    s = tl.where(miss, tl.load(slot_ptr + k), 0)
    wty = tl.bfloat16 if W_BF16 else tl.float16
    if p == 0:
        src_a = tl.load(host_g + layer_base + e)
        dst_a = tl.load(slot_g + s)
        n = n_g
    elif p == 1:
        src_a = tl.load(host_u + layer_base + e)
        dst_a = tl.load(slot_u + s)
        n = n_u
    else:
        src_a = tl.load(host_d + layer_base + e)
        dst_a = tl.load(slot_d + s)
        n = n_d
    src = tl.cast(src_a, tl.pointer_type(wty), bitcast=True)
    dst = tl.cast(dst_a, tl.pointer_type(wty), bitcast=True)
    offs = c * BLOCK + tl.arange(0, BLOCK)
    m = (offs < n) & miss
    tl.store(dst + offs, tl.load(src + offs, mask=m), mask=m)
