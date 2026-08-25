"""Triton MoE Fused Dispatch — zero torch, zero extra memory.

Fused Grouped GEMM for Mixture of Experts. One kernel launch per projection
handles ALL experts instead of iterating one-by-one.

Architecture:
  1. moe_align_block_size: sort tokens by expert, pad to BLOCK_M alignment
  2. Build offset tables: [E] int64 element offsets into arena (zero-copy, cached)
  3. Pass 1 (gate): fused GEMM via offset table → [padded, N]
  4. Pass 2 (up):   fused GEMM via offset table → [padded, N]
  5. Activation:    silu(gate) * up → [padded, N]
  6. Pass 3 (down): fused GEMM + routing weights → [padded, K]
  7. Reduce across top_k → [M, K]

Zero extra GPU memory — expert weights stay in the arena. Only a small
offset table ([E] int64 = 1KB for 128 experts) is allocated per projection.

Multi-GPU: one set of kernel launches per device, bulk D2D transfers.
"""

import ctypes as _ctypes
import os as _os
from collections import defaultdict, OrderedDict

import numpy as np

from neurobrix.kernels import wrappers as w
from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator, dtype_size

# NBX_MOE_DIAG gate, hoisted to import time (C1 hygiene: never read the
# environ on the per-MoE-op hot path). Empty/unset → falsy → zero cost.
# A non-"1" value doubles as a cache_key substring filter for the
# intermediate dumps (NBX_DUMP_TIDS_FILTER idiom); "1" dumps every block.
_MOE_DIAG_ENV = _os.environ.get("NBX_MOE_DIAG", "")

# Block size for token alignment (must match wrapper's _MOE_BM)
_BLOCK_SIZE_M = 16



# ============================================================================
# POINTER TABLE BUILDER — zero-copy, cached
# ============================================================================

# LRU cache of pointer tables keyed by a data_ptr fingerprint of every
# expert weight. Under zero3 pipelining weights may be freed and
# reallocated at the same virtual address, so a static cache_key
# (e.g. first expert's data_ptr) can resolve to a stale PtrTables and
# cause silent garbage or illegal memory accesses. Hashing ALL
# expert data_ptrs detects any change to the arena layout.
#
# Bounded by _PTR_CACHE_MAXSIZE entries — eviction is LRU (OrderedDict
# move_to_end on hit, popitem(last=False) on overflow). The cap keeps
# memory flat when running many blocks across devices during
# pipelining; 256 entries × 3 int64 tables × few hundred experts is
# still sub-megabyte.
_PTR_CACHE_MAXSIZE = 256
_ptr_cache: "OrderedDict[int, PtrTables]" = OrderedDict()


def _ptr_cache_fingerprint(
    gate_weights, up_weights, down_weights, num_experts: int
) -> int:
    """Stable hash over every expert's data_ptr for all three projections.

    Any weight swap (H2D, D2D, free-and-remalloc at the same address)
    changes at least one data_ptr — and in the rare adversarial case
    where the driver hands out identical addresses for a pair of
    identically-sized buffers, we still hash all 3 × num_experts ptrs
    so a single match isn't enough for a false cache hit.
    """
    vals = []
    vals.append(num_experts)
    for i in range(num_experts):
        vals.append(gate_weights[i].data_ptr())
        vals.append(up_weights[i].data_ptr())
        vals.append(down_weights[i].data_ptr())
    return hash(tuple(vals))


def _ptr_cache_get(fp: int):
    """LRU lookup. Promotes the hit to the MRU end."""
    tables = _ptr_cache.get(fp)
    if tables is not None:
        _ptr_cache.move_to_end(fp)
    return tables


def _ptr_cache_put(fp: int, tables):
    """Insert + evict the oldest entry if we're over the cap."""
    _ptr_cache[fp] = tables
    _ptr_cache.move_to_end(fp)
    while len(_ptr_cache) > _PTR_CACHE_MAXSIZE:
        _ptr_cache.popitem(last=False)


class PtrTables:
    """Per-projection absolute pointer tables for zero-copy expert access.

    Each table is [E] int64 on GPU: absolute data_ptr() of each expert.
    The fused kernel loads ptrs[expert_id] and uses it directly as a pointer.
    """
    __slots__ = ('gate_ptrs', 'gate_stride_bk', 'gate_stride_bn',
                 'up_ptrs', 'up_stride_bk', 'up_stride_bn',
                 'down_ptrs', 'down_stride_bk', 'down_stride_bn',
                 'device_experts', 'quantized', 'q_tables')

    def __init__(self):
        self.gate_ptrs = {}       # {device → NBXTensor[E] int64}
        self.gate_stride_bk = {}
        self.gate_stride_bn = {}
        self.up_ptrs = {}
        self.up_stride_bk = {}
        self.up_stride_bn = {}
        self.down_ptrs = {}
        self.down_stride_bk = {}
        self.down_stride_bn = {}
        self.device_experts = {}
        # int4-g128-asym experts: *_ptrs holds the qweight table and
        # q_tables[proj][dev] = (sc_ptrs, mn_ptrs, stride_sg, stride_sn);
        # the *_stride_bk/bn slots hold the PACKED int32 strides.
        self.quantized = False
        self.q_tables = {}


def _build_ptr_tables(gate_weights, up_weights, down_weights):
    """Build per-projection absolute pointer tables.

    Stores each expert's data_ptr() as int64 in a GPU tensor. No offset math.
    Total GPU allocation: 3 × E × 8 bytes per device (~3KB for 128 experts).
    """
    from neurobrix.kernels.quantized_tensor import QuantizedTensor

    num_experts = len(gate_weights)
    tables = PtrTables()
    tables.quantized = isinstance(gate_weights[0], QuantizedTensor)
    if tables.quantized:
        # Uniformity contract: mixed dense/quantized expert lists are a
        # broken build — refuse loudly (ZERO FALLBACK).
        for proj_name, ws in (("gate", gate_weights), ("up", up_weights),
                              ("down", down_weights)):
            if not all(isinstance(x, QuantizedTensor) for x in ws):
                raise RuntimeError(
                    f"ZERO FALLBACK: mixed dense/quantized experts in "
                    f"'{proj_name}' projection — re-build the variant.")
    by_device = defaultdict(list)
    for i in range(num_experts):
        by_device[gate_weights[i]._device_idx].append(i)
    tables.device_experts = dict(by_device)

    for dev, expert_ids in by_device.items():
        DeviceAllocator.set_device(dev)

        def _build_proj(weights):
            # Absolute pointers as int64
            if tables.quantized:
                qw = np.array(
                    [weights[eid].qweight.data_ptr() for eid in expert_ids],
                    dtype=np.int64)
                sc = np.array(
                    [weights[eid].scales.data_ptr() for eid in expert_ids],
                    dtype=np.int64)
                mn = np.array(
                    [weights[eid].qmins.data_ptr() for eid in expert_ids],
                    dtype=np.int64)
                base = weights[expert_ids[0]]
                return ((NBXTensor.from_numpy(qw),
                         NBXTensor.from_numpy(sc),
                         NBXTensor.from_numpy(mn)),
                        base.qweight.stride(0), base.qweight.stride(1),
                        base.scales.stride(0), base.scales.stride(1))
            ptrs = np.array(
                [weights[eid].data_ptr() for eid in expert_ids],
                dtype=np.int64)
            ptr_table = NBXTensor.from_numpy(ptrs)
            base = weights[expert_ids[0]]
            return ptr_table, base.stride(1), base.stride(0)

        if tables.quantized:
            for name, ws in (("gate", gate_weights), ("up", up_weights),
                             ("down", down_weights)):
                (qw_t, sc_t, mn_t), qk, qn, sg, sn = _build_proj(ws)
                getattr(tables, f"{name}_ptrs")[dev] = qw_t
                getattr(tables, f"{name}_stride_bk")[dev] = qk
                getattr(tables, f"{name}_stride_bn")[dev] = qn
                tables.q_tables.setdefault(name, {})[dev] = (
                    sc_t, mn_t, sg, sn)
            continue

        gp, gbk, gbn = _build_proj(gate_weights)
        tables.gate_ptrs[dev] = gp
        tables.gate_stride_bk[dev] = gbk
        tables.gate_stride_bn[dev] = gbn

        up, ubk, ubn = _build_proj(up_weights)
        tables.up_ptrs[dev] = up
        tables.up_stride_bk[dev] = ubk
        tables.up_stride_bn[dev] = ubn

        dp, dbk, dbn = _build_proj(down_weights)
        tables.down_ptrs[dev] = dp
        tables.down_stride_bk[dev] = dbk
        tables.down_stride_bn[dev] = dbn

    return tables


# ============================================================================
# MOE ALIGN BLOCK SIZE — device-side token-by-expert sort (3 Triton kernels)
# ============================================================================

def moe_align_block_size(topk_ids_flat, block_size, num_experts, device_idx):
    """Sort tokens by expert, pad to block_size alignment — ON DEVICE.

    Three constant-tuple Triton kernels (kernels/ops/moe_align.py)
    replace the former host path (D2H of topk_ids -> numpy sort -> H2D
    of tables): the routing tables are now recomputed from the CURRENT
    router output at every launch, which removes one D2H sync per MoE
    layer per step AND makes the MoE band replay-eligible (the host
    D2H was a structural plan-breaker at recording — P-REPLAY-KV-
    DECODE). The deterministic pairwise rank in stage 3 preserves the
    host sort's exact token order — the swap is byte-gated against the
    removed implementation.

    Outputs are sized on the shape-derived worst case
    min(n*BS, n + E*(BS-1)) rounded up to BS; the TRUE total lives only
    in the device scalar (every fused variant early-exits on it — the
    launch grids depend on host shapes alone, never on routing data).

    Args:
        topk_ids_flat: NBXTensor [M * top_k] — flat expert indices
        block_size: BLOCK_SIZE_M for the fused kernel
        num_experts: total number of experts
        device_idx: GPU device for output tensors

    Returns:
        sorted_token_ids: NBXTensor [max_total] — token indices sorted
            by expert (sentinel n beyond each expert's true count)
        expert_ids: NBXTensor [max_blocks] — expert id per block (-1 pad)
        num_tokens_post_padded: NBXTensor [1] — total entries after padding
    """
    import triton as _tr
    from neurobrix.kernels.nbx_tensor import _set_device
    from neurobrix.kernels.ops.moe_align import (
        moe_align_stage1_kernel, moe_align_stage2_kernel,
        moe_align_stage3_kernel)

    n = topk_ids_flat.numel()
    bs = int(block_size)
    bound = min(n * bs, n + num_experts * (bs - 1))
    max_total = ((bound + bs - 1) // bs) * bs
    max_blocks = max_total // bs

    DeviceAllocator.set_device(device_idx)
    dev = f"cuda:{device_idx}"
    sorted_ids = NBXTensor.empty((max_total,), dtype=NBXDtype.int64,
                                 device=dev)
    expert_ids = NBXTensor.empty((max_blocks,), dtype=NBXDtype.int64,
                                 device=dev)
    num_post_pad = NBXTensor.empty((1,), dtype=NBXDtype.int64, device=dev)
    offsets_ws = NBXTensor.empty((num_experts,), dtype=NBXDtype.int64,
                                 device=dev)
    padded_ws = NBXTensor.empty((num_experts,), dtype=NBXDtype.int64,
                                device=dev)

    ids = topk_ids_flat.contiguous()
    _set_device(sorted_ids)
    BE = max(_tr.next_power_of_2(num_experts), 16)
    BLK = ((128 + bs - 1) // bs) * bs  # positions/program, multiple of bs
    moe_align_stage1_kernel[(1,)](
        ids, offsets_ws, padded_ws, num_post_pad, n,
        BS=bs, BE=BE, E=num_experts, BT=128, num_warps=4)
    moe_align_stage2_kernel[(_tr.cdiv(max_total, BLK),)](
        offsets_ws, padded_ws, sorted_ids, expert_ids, n, max_total,
        BS=bs, BE=BE, E=num_experts, BLK=BLK, num_warps=4)
    moe_align_stage3_kernel[(_tr.cdiv(n, 128),)](
        ids, offsets_ws, sorted_ids, n,
        E=num_experts, BLKT=128, BN=128, num_warps=4)

    return sorted_ids, expert_ids, num_post_pad


# ============================================================================
# D2D TRANSFER HELPER
# ============================================================================

def _xfer(tensor: NBXTensor, target_dev: int) -> NBXTensor:
    """Move an NBXTensor to target_dev via the shared cross-device brick.

    Same-device: no-op. Cross-device: `device_transfer.transfer_tensor`,
    which enforces the D2D read-ordering contract (source-device sync
    barrier before the peer copy) and materialises non-dense views on
    the source device. The previous local flat memcpy skipped BOTH — a
    D2D enqueued on the target's legacy stream never waits for the
    source device's stream, so routing metadata / activations could be
    copied while still being written: layout-dependent illegal-address
    poison at block-scatter boundaries, surfacing at the fused-MoE
    launch of the first block on the next device (D9 layer 2; same
    class as the Qwen3-Omni audio step-2 precedent recorded in
    transfer_tensor's docstring).
    """
    if tensor._device_idx == target_dev:
        return tensor
    from neurobrix.triton.device_transfer import transfer_tensor
    return transfer_tensor(tensor, target_dev)


# ============================================================================
# MAIN ENTRY POINT — FUSED EXECUTION
# ============================================================================

def execute_moe_fused(
    gate_scores: NBXTensor,
    hidden_states: NBXTensor,
    gate_weights, up_weights, down_weights,
    top_k: int,
    num_experts: int,
    norm_topk_prob: bool = True,
    cache_key: str = "",
    topk_indices=None,
    topk_weights=None,
):
    """Execute MoE via fused grouped GEMM — zero torch, zero extra memory.

    API compatible with old per-expert loop. Internally uses fused Triton
    kernels with zero-copy offset tables for expert weight access.

    Args:
        gate_scores: Router logits [batch*seq, num_experts]. None when the
            routing is pre-computed (multi-gate blend, see below).
        hidden_states: Input activations [batch*seq, hidden_dim]
        gate_weights: List of gate weight NBXTensors per expert
        up_weights: List of up weight NBXTensors per expert
        down_weights: List of down weight NBXTensors per expert
        top_k: Number of experts per token
        num_experts: Total number of experts
        norm_topk_prob: Whether to normalize routing probabilities. IGNORED
            when topk_indices/topk_weights are supplied — that routing is
            already normalized per gate inside the graph.
        cache_key: Stable key for offset table caching (component + op_uid)
        topk_indices: Pre-computed expert indices [batch*seq, top_k], int.
            Multi-gate MoE (e.g. BailingMoe text/image/audio routers) blends
            several gates' topk results by per-token modality masks IN THE
            GRAPH; the blend result is bound by moe_fusion.py and consumed
            here verbatim. Mirror of the compiled engine (R30).
        topk_weights: Pre-computed routing weights [batch*seq, top_k], float.
            Supplied together with topk_indices or not at all.
    """
    if hidden_states is None:
        raise RuntimeError("MoE fused: hidden_states is None")

    _blended = topk_indices is not None or topk_weights is not None
    if _blended and (topk_indices is None or topk_weights is None):
        raise RuntimeError(
            "MoE fused (multi-gate): topk_indices and topk_weights must be "
            "supplied together — got "
            f"indices={'None' if topk_indices is None else 'OK'}, "
            f"weights={'None' if topk_weights is None else 'OK'}")
    if not _blended and gate_scores is None:
        raise RuntimeError("MoE fused: gate_scores is None and no blended routing")

    # Set device context to activation device
    act_dev = hidden_states._device_idx
    DeviceAllocator.set_device(act_dev)
    DeviceAllocator.ensure_triton_device(act_dev)

    # Transfer routing tensors to activation device if needed
    if _blended:
        if topk_indices._device_idx != act_dev:
            topk_indices = _xfer(topk_indices, act_dev)
        if topk_weights._device_idx != act_dev:
            topk_weights = _xfer(topk_weights, act_dev)
    elif gate_scores._device_idx != act_dev:
        gate_scores = _xfer(gate_scores, act_dev)

    # Zero3 CPU offload: expert weights may be on pinned host memory.
    # They MUST be on act_dev before _build_ptr_tables runs because the
    # pointer table caches raw data_ptrs — a CPU ptr in a GPU int64
    # table dereferenced inside the kernel would crash or corrupt.
    # This promotion re-creates GPU NBXTensors via to_cuda() which
    # issues H2D cudaMemcpy. Expected to fire on every MoE op under
    # zero3 (the CPU→GPU copy IS the zero3 slow path for MoE).
    #
    # Cache invalidation: if ANY weight is CPU, we bypass _ptr_cache
    # entirely for this call and build a fresh table from the promoted
    # tensors. Caching the table when weights were CPU would bake in
    # GPU pointers that become stale on the NEXT call (when the CPU
    # originals are re-promoted and land at different GPU addresses,
    # or when future pipelining evicts the GPU copy). Under pure GPU
    # multi-device setups the cache works normally and gives its usual
    # ~3 KB-per-call saving; the zero3 path accepts the rebuild cost
    # as the tradeoff for correct per-call placement.
    #
    # DEFERRED: A zero3-pipelining-aware cache will need to key on a
    # version counter that the ratchet bumps on every eviction, so
    # resident blocks keep a stable cache entry while evicted ones
    # force a rebuild. See tests/scratch/zero3_leak_investigation/
    # REPORT.md for the groundwork APIs.
    any_cpu_weight = any(
        getattr(w, '_device', 'cuda') == 'cpu'
        for lst in (gate_weights, up_weights, down_weights)
        for w in lst)
    import os as _os_z3
    _Z3_DIAG = _os_z3.environ.get("NBX_Z3_TRITON_DIAG") == "1"
    if _Z3_DIAG and any_cpu_weight:
        print(f"[Z3_MOE] {cache_key} pre-promote "
              f"cuda_live={DeviceAllocator.memory_allocated(act_dev)/1e6:.1f}MB",
              flush=True)
    if any_cpu_weight:
        gate_weights = [
            w.to_cuda(act_dev) if getattr(w, '_device', 'cuda') == 'cpu' else w
            for w in gate_weights]
        up_weights = [
            w.to_cuda(act_dev) if getattr(w, '_device', 'cuda') == 'cpu' else w
            for w in up_weights]
        down_weights = [
            w.to_cuda(act_dev) if getattr(w, '_device', 'cuda') == 'cpu' else w
            for w in down_weights]
    if _Z3_DIAG and any_cpu_weight:
        print(f"[Z3_MOE] {cache_key} post-promote "
              f"cuda_live={DeviceAllocator.memory_allocated(act_dev)/1e6:.1f}MB",
              flush=True)

    orig_shape = hidden_states.shape
    if hidden_states.ndim == 3:
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    if _blended:
        if topk_indices.ndim > 2:
            topk_indices = topk_indices.reshape(-1, topk_indices.shape[-1])
        if topk_weights.ndim > 2:
            topk_weights = topk_weights.reshape(-1, topk_weights.shape[-1])
    elif gate_scores.ndim == 3:
        gate_scores = gate_scores.reshape(-1, gate_scores.shape[-1])

    # Resolve weight dtype
    w_dtype = gate_weights[0]._dtype
    if hidden_states._dtype != w_dtype:
        hidden_states = hidden_states.to(w_dtype)

    M, K = hidden_states.shape
    N_gate = gate_weights[0].shape[0]  # intermediate_dim (gate/up are [intermediate, hidden])

    # Env-gated MoE intermediate dumps (§8 NBX_MOE_DIAG; gate hoisted to
    # module import — zero environ reads on this hot path).
    _moe_diag = _MOE_DIAG_ENV
    _moe_diag_cache_key = cache_key
    def _dump(label, tensor):
        if not _moe_diag:
            return
        # Value other than "1" = cache_key substring filter
        # (NBX_DUMP_TIDS_FILTER idiom).
        if _moe_diag != "1" and _moe_diag not in _moe_diag_cache_key:
            return
        try:
            import ctypes as _ct
            from neurobrix.kernels.nbx_tensor import NBXDtype as _NBX
            DeviceAllocator.set_device(tensor._device_idx)
            full = tensor if tensor.dtype == _NBX.float32 else tensor.to(_NBX.float32)
            full = full.contiguous()
            n = full.numel()
            buf = (_ct.c_float * n)()
            DeviceAllocator.memcpy(_ct.addressof(buf), full.data_ptr(), n * 4, kind=2)
            vals = list(buf)
            head = vals[:10]
            norm = (sum(v * v for v in vals)) ** 0.5
            import sys as _sys
            print(f"[MOE_DIAG] {label:20s} shape={list(tensor.shape)} "
                  f"dtype={tensor.dtype} norm={norm:.4f} head10={[round(v,4) for v in head]}",
                  file=_sys.stderr, flush=True)
        except Exception as _e:
            import sys as _sys
            print(f"[MOE_DIAG] {label} failed: {_e}", file=_sys.stderr, flush=True)
    if not _blended:
        _dump("gate_scores_in", gate_scores)
    _dump("hidden_states_in", hidden_states)
    # =========================================

    # ================================================================
    # STEP 1: Routing — topk + normalize, OR pre-computed blend
    # ================================================================
    if _blended:
        # Multi-gate: the graph already ran N gates (softmax → topk →
        # normalize) and blended them by per-token modality masks. Re-running
        # topk here would discard the modality selection; re-normalizing
        # would double-normalize. Consume verbatim.
        indices = topk_indices
        if indices._dtype != NBXDtype.int64:
            # moe_align_block_size and _multi_device_fused_moe read the flat
            # index buffer as int64 via ctypes — normalize at the boundary.
            indices = indices.to(NBXDtype.int64)
        scores = topk_weights.to(NBXDtype.float32)
        _dump("blend_indices", indices)
        _dump("blend_scores", scores)
    else:
        gate_fp32 = gate_scores.to(NBXDtype.float32)
        scores, indices = w.topk_wrapper(gate_fp32, top_k, dim=-1)
        _dump("topk_scores", scores)
        _dump("topk_indices", indices)

        if norm_topk_prob:
            denom = w.sum_wrapper(scores, dim=-1, keepdim=True)
            scores = w.div(scores, denom)
            _dump("scores_norm", scores)

    # Flatten routing for fused kernel
    flat_indices = indices.reshape(-1)                 # [M * top_k]
    flat_scores = scores.reshape(-1).to(w_dtype)       # [M * top_k]
    _dump("flat_scores", flat_scores)

    # ================================================================
    # STEP 2: Build pointer tables (cached — zero-copy)
    # ================================================================
    # Under zero3 (any_cpu_weight=True above), we skip the cache: the
    # promoted GPU tensors are freshly allocated per call, so cached
    # pointers would dangle after this call returns. See comment at
    # STEP 0 for the zero3/pipelining interaction.
    #
    # Under pipelining (all weights already on GPU but evicted/promoted
    # between blocks), the cache key is a fingerprint of EVERY expert
    # data_ptr so a swapped buffer invalidates the cache. LRU-bounded
    # to _PTR_CACHE_MAXSIZE entries.
    if any_cpu_weight:
        tables = _build_ptr_tables(gate_weights, up_weights, down_weights)
        DeviceAllocator.set_device(act_dev)
        DeviceAllocator.ensure_triton_device(act_dev)
    else:
        fp = _ptr_cache_fingerprint(
            gate_weights, up_weights, down_weights, num_experts)
        tables = _ptr_cache_get(fp)
        if tables is None:
            tables = _build_ptr_tables(
                gate_weights, up_weights, down_weights)
            _ptr_cache_put(fp, tables)
            DeviceAllocator.set_device(act_dev)
            DeviceAllocator.ensure_triton_device(act_dev)

    # ================================================================
    # STEP 2b: SIMT decode band — M == 1, int4-g128 experts (the
    # fourth SIMT family member, 2026-08-24). At one token the grouped
    # GEMM's every 16-row tile carries ONE real row and the sorted-
    # token machinery sorts a list of top_k entries; this path replaces
    # the whole band (align + 3 grouped launches + SwiGLU pass + sum
    # combine) with two kernels whose cross-expert combine is a FIXED-
    # ORDER in-kernel loop (see ops/moe_decode_vec.py). Capability-
    # gated, data-driven: quantized tables + M==1 + single device +
    # the g128/int4 shape contract verified from the weights — every
    # other MoE (fp16 experts, prefill M>1, multi-device) keeps the
    # proven path below. NBX_MOE_VEC three-state: "1"
    # armed explicitly, unset = DEFAULT (ADOPTED 2026-08-25: locked
    # 1290, interleaved n=5, no overlap — short +27.6% (the row's
    # first 30+ tok/s: 31.07), 4,164 +24.1%, ~8,300 +20.8%; short and
    # long byte-identical, xlong moved ONCE pass-stable under the new
    # fixed reduction order; judged config BN=64/BK=128/W=4), "0" kill.
    import os as _os_mv
    if (M == 1 and tables.quantized
            and len(tables.device_experts) == 1
            and _os_mv.environ.get("NBX_MOE_VEC", "1") != "0"):
        _q0 = gate_weights[0]
        _group = (K // _q0.scales.shape[0]
                  if getattr(_q0, "scales", None) is not None
                  and _q0.scales.shape[0] > 0 else 0)
        _down0 = down_weights[0]
        _dgroup = (N_gate // _down0.scales.shape[0]
                   if getattr(_down0, "scales", None) is not None
                   and _down0.scales.shape[0] > 0 else 0)
        if _group == 128 and _dgroup == 128:
            dev = next(iter(tables.device_experts))
            if dev != act_dev:
                DeviceAllocator.set_device(dev)
                DeviceAllocator.ensure_triton_device(dev)
                hidden_states = _xfer(hidden_states, dev)
                flat_scores = _xfer(flat_scores, dev)
                flat_indices = _xfer(flat_indices, dev)
            output = _moe_decode_vec_pass(
                hidden_states, tables, dev, flat_indices, flat_scores,
                top_k, K, N_gate)
            if dev != act_dev:
                output = _xfer(output, act_dev)
                DeviceAllocator.set_device(act_dev)
                DeviceAllocator.ensure_triton_device(act_dev)
            if len(orig_shape) == 3:
                output = output.reshape(orig_shape)
            DeviceAllocator.set_device(act_dev)
            DeviceAllocator.ensure_triton_device(act_dev)
            return output

    # ================================================================
    # STEP 3: Align tokens by expert (sorting)
    # ================================================================
    sorted_token_ids, expert_ids, num_tokens_post_padded = \
        moe_align_block_size(flat_indices, _BLOCK_SIZE_M, num_experts, act_dev)

    # ================================================================
    # STEP 4: Per-device fused execution
    # ================================================================
    all_same_device = len(tables.device_experts) == 1

    if all_same_device:
        dev = next(iter(tables.device_experts))
        if dev != act_dev:
            DeviceAllocator.set_device(dev)
            DeviceAllocator.ensure_triton_device(dev)
            hidden_states = _xfer(hidden_states, dev)
            flat_scores = _xfer(flat_scores, dev)
            sorted_token_ids = _xfer(sorted_token_ids, dev)
            expert_ids = _xfer(expert_ids, dev)
            num_tokens_post_padded = _xfer(num_tokens_post_padded, dev)

        if _moe_diag:
            import sys as _sys_d
            print(f"[MOE_DIAG] {cache_key} dev={dev} act_dev={act_dev} "
                  f"h=(d{hidden_states._device_idx},{hidden_states.shape},{hidden_states.data_ptr():#x}) "
                  f"scores=(d{flat_scores._device_idx},{flat_scores.shape}) "
                  f"sorted=(d{sorted_token_ids._device_idx},{sorted_token_ids.shape}) "
                  f"expids=(d{expert_ids._device_idx},{expert_ids.shape}) "
                  f"npad=(d{num_tokens_post_padded._device_idx}) "
                  f"tbl_gate=(d{tables.gate_ptrs[dev]._device_idx},{tables.gate_ptrs[dev].shape},{tables.gate_ptrs[dev].data_ptr():#x}) "
                  f"M={M} topk={top_k}", file=_sys_d.stderr, flush=True)

        result = _fused_moe_pass(
            hidden_states, tables, dev, flat_scores,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            top_k, M, K, N_gate,
            _diag_dump=_dump if _moe_diag else None,
        )

        if dev != act_dev:
            result = _xfer(result, act_dev)
            DeviceAllocator.set_device(act_dev)
            DeviceAllocator.ensure_triton_device(act_dev)

        output = result
    else:
        output = _multi_device_fused_moe(
            hidden_states, flat_indices, flat_scores,
            tables, gate_weights, up_weights, down_weights,
            top_k, num_experts, M, K, N_gate, act_dev,
        )

    if len(orig_shape) == 3:
        output = output.reshape(orig_shape)

    # Under zero3 (any_cpu_weight=True), explicitly drop the promoted
    # weight lists AND the local `tables` before returning. The ptr
    # tables' int64 addresses become stale once the GPU NBXTensors are
    # freed, so we must release both in the same frame. Without this,
    # CPython may keep the function frame alive one extra tick on the
    # caller's stack, holding ~800 MB per MoE op and OOMing after 7-8
    # blocks on a 16 GB V100.
    if any_cpu_weight:
        del gate_weights, up_weights, down_weights
        del tables
        import gc as _gc
        _gc.collect()

    DeviceAllocator.set_device(act_dev)
    DeviceAllocator.ensure_triton_device(act_dev)

    if _Z3_DIAG and any_cpu_weight:
        print(f"[Z3_MOE] {cache_key} post-return "
              f"cuda_live={DeviceAllocator.memory_allocated(act_dev)/1e6:.1f}MB",
              flush=True)

    return output


def _moe_decode_vec_pass(hidden_states, tables, dev, flat_indices,
                         flat_scores, top_k, K, N_gate):
    """Three-launch SIMT decode band (see ops/moe_decode_vec.py):
    gateup + down-split (expert axis on the grid) + fixed-order part
    reduce. NBX_MOEV_DSPLIT=0 restores the first-pass fused down
    combine.

    hidden_states [1, K] fp16 · flat_indices [top_k] int64 ·
    flat_scores [top_k]. Returns [1, K] in hidden's dtype.
    """
    import os as _os_dv
    import triton as _tr
    from neurobrix.kernels.ops.moe_decode_vec import (
        moe_gateup_vec_kernel, moe_down_combine_vec_kernel,
        moe_down_split_vec_kernel, moe_part_reduce_kernel)
    from neurobrix.kernels.nbx_tensor import _set_device

    dt = hidden_states._dtype
    x = hidden_states.reshape(K)
    h = NBXTensor.empty((top_k, N_gate), dtype=NBXDtype.float32,
                        device=f"cuda:{dev}")
    out32 = NBXTensor.empty((K,), dtype=NBXDtype.float32,
                            device=f"cuda:{dev}")

    g_sc, g_mn, sg, sn = tables.q_tables["gate"][dev]
    u_sc, u_mn, _, _ = tables.q_tables["up"][dev]
    d_sc, d_mn, dsg, dsn = tables.q_tables["down"][dev]

    # Adopted second-pass grid (2026-08-25): BN=32 is the gateup tile
    # from the profile ladder (71 us/call vs 87.5 at BN=64; 16/BK256/
    # W2/W8 all slower). The down projection moves its expert axis to
    # the grid — 512 programs vs the fused loop's 32 on 80 SMs
    # (46 -> ~248 GB/s) — with a fixed-order part reduce completing
    # the deterministic combine.
    BN = int(_os_dv.environ.get("NBX_MOEV_BN", "32"))
    BK = int(_os_dv.environ.get("NBX_MOEV_BK", "128"))
    W = int(_os_dv.environ.get("NBX_MOEV_WARPS", "4"))
    BND = int(_os_dv.environ.get("NBX_MOEV_BND", str(BN)))
    WD = int(_os_dv.environ.get("NBX_MOEV_WARPS_D", str(W)))

    _set_device(h)
    moe_gateup_vec_kernel[(top_k, _tr.cdiv(N_gate, BN))](
        x, flat_indices,
        tables.gate_ptrs[dev], g_sc, g_mn,
        tables.up_ptrs[dev], u_sc, u_mn,
        h, K, N_gate,
        tables.gate_stride_bk[dev], tables.gate_stride_bn[dev],
        sg, sn,
        BLOCK_N=BN, BLOCK_K=BK, GROUP=128, PACK=8,
        num_warps=W, num_stages=1)
    if _os_dv.environ.get("NBX_MOEV_DSPLIT", "1") != "0":
        part = NBXTensor.empty((top_k, K), dtype=NBXDtype.float32,
                               device=f"cuda:{dev}")
        moe_down_split_vec_kernel[(top_k, _tr.cdiv(K, BND))](
            h, flat_indices, flat_scores,
            tables.down_ptrs[dev], d_sc, d_mn,
            part, N_gate, K,
            tables.down_stride_bk[dev], tables.down_stride_bn[dev],
            dsg, dsn,
            BLOCK_N=BND, BLOCK_K=BK, GROUP=128, PACK=8,
            num_warps=WD, num_stages=1)
        moe_part_reduce_kernel[(_tr.cdiv(K, 256),)](
            part, out32, K, TOP_K=top_k, BLOCK_N=256, num_warps=2)
        return out32.to(dt).reshape(1, K)
    moe_down_combine_vec_kernel[(_tr.cdiv(K, BND),)](
        h, flat_indices, flat_scores,
        tables.down_ptrs[dev], d_sc, d_mn,
        out32, N_gate, K,
        tables.down_stride_bk[dev], tables.down_stride_bn[dev],
        dsg, dsn,
        TOP_K=top_k, BLOCK_N=BND, BLOCK_K=BK, GROUP=128, PACK=8,
        num_warps=WD, num_stages=1)
    return out32.to(dt).reshape(1, K)


# ============================================================================
# FUSED MOE PASS (three GEMMs + activation, zero-copy)
# ============================================================================

def _fused_moe_pass(
    hidden_states, tables, dev, flat_scores,
    sorted_token_ids, expert_ids, num_tokens_post_padded,
    top_k, M, K, N_gate,
    _diag_dump=None,
):
    """Three fused GEMM passes + silu activation."""
    dt = hidden_states._dtype
    total_tokens = M * top_k
    padded = sorted_token_ids.shape[0]

    # Diagnostic: dump the stride config being passed to the kernel for each
    # projection. DeepSeek 1408/2048 layout vs Qwen3 1536/2048 — a wrong
    # stride_bk/stride_bn swap would explain the 30× magnitude hit.
    if _diag_dump is not None:
        import sys as _sys
        print(f"[MOE_DIAG] strides gate: bk={tables.gate_stride_bk[dev]} "
              f"bn={tables.gate_stride_bn[dev]}", file=_sys.stderr, flush=True)
        print(f"[MOE_DIAG] strides up:   bk={tables.up_stride_bk[dev]} "
              f"bn={tables.up_stride_bn[dev]}", file=_sys.stderr, flush=True)
        print(f"[MOE_DIAG] strides down: bk={tables.down_stride_bk[dev]} "
              f"bn={tables.down_stride_bn[dev]}", file=_sys.stderr, flush=True)
        print(f"[MOE_DIAG] sorted_token_ids.numel={sorted_token_ids.numel()} "
              f"expert_ids.numel={expert_ids.numel()} M={M} top_k={top_k} "
              f"N_gate={N_gate} K={K}", file=_sys.stderr, flush=True)

    def _proj(name, x, out, N_p, K_p, mul_routed, topk_div=True):
        """One grouped-GEMM pass — dense or W4 per the tables."""
        if tables.quantized:
            sc_t, mn_t, sg, sn = tables.q_tables[name][dev]
            w.invoke_fused_moe_wna16(
                x, getattr(tables, f"{name}_ptrs")[dev], sc_t, mn_t,
                out, flat_scores,
                sorted_token_ids, expert_ids, num_tokens_post_padded,
                N_p, K_p,
                getattr(tables, f"{name}_stride_bk")[dev],
                getattr(tables, f"{name}_stride_bn")[dev],
                sg, sn,
                top_k, mul_routed_weight=mul_routed, topk_divide=topk_div,
            )
        else:
            w.invoke_fused_moe(
                x, getattr(tables, f"{name}_ptrs")[dev], out, flat_scores,
                sorted_token_ids, expert_ids, num_tokens_post_padded,
                N_p, K_p,
                getattr(tables, f"{name}_stride_bk")[dev],
                getattr(tables, f"{name}_stride_bn")[dev],
                top_k, mul_routed_weight=mul_routed, topk_divide=topk_div,
            )

    gate_out = NBXTensor.zeros((padded, N_gate), dtype=dt, device=f"cuda:{dev}")
    _proj("gate", hidden_states, gate_out, N_gate, K, False)
    if _diag_dump is not None:
        _diag_dump("gate_out", gate_out)

    up_out = NBXTensor.zeros((padded, N_gate), dtype=dt, device=f"cuda:{dev}")
    _proj("up", hidden_states, up_out, N_gate, K, False)
    if _diag_dump is not None:
        _diag_dump("up_out", up_out)

    # SwiGLU: silu(gate) * up
    gate_silu = w.silu(gate_out)
    activated = w.mul(gate_silu, up_out)
    if _diag_dump is not None:
        _diag_dump("activated", activated)

    # Down pass
    down_out = NBXTensor.zeros((padded, K), dtype=dt, device=f"cuda:{dev}")
    _proj("down", activated, down_out, K, N_gate, True, topk_div=False)
    if _diag_dump is not None:
        _diag_dump("down_out", down_out)

    # Reduce across top_k
    result = down_out.narrow(0, 0, total_tokens).reshape(M, top_k, K)
    result = w.sum_wrapper(result, dim=1)
    if _diag_dump is not None:
        _diag_dump("result_final", result)
    return result


# ============================================================================
# MULTI-GPU DISPATCH
# ============================================================================

def _multi_device_fused_moe(
    hidden_states, flat_indices, flat_scores,
    tables, _gate_weights, _up_weights, _down_weights,
    top_k, _num_experts, M, K, N_gate, act_dev,
):
    """Execute fused MoE across multiple GPUs.

    For each device holding experts:
    1. Identify which flat routing positions go to experts on this device
    2. Build GLOBAL sorted_token_ids (original flat positions, not remapped)
    3. Transfer activations + scores subset to device
    4. Run fused kernel with per-device offset tables
    5. Scatter results back using original token positions
    """
    dt = hidden_states._dtype
    # (M + 1, K): row M is a discard SINK. The padding sentinel
    # (sorted_tids == n_total) maps to token_id = n_total // top_k = M
    # (since n_total = M * top_k), so the device-side index_add routes every
    # padding row into the sink with no explicit mask. Narrowed back to
    # (M, K) after the device loop. (D7 Stage B.)
    output = NBXTensor.zeros((M + 1, K), dtype=dt, device=f"cuda:{act_dev}")

    n_total = flat_indices.numel()

    # D2H the flat indices to know which expert each token goes to
    buf = (_ctypes.c_int64 * n_total)()
    DeviceAllocator.memcpy(_ctypes.addressof(buf), flat_indices.data_ptr(),
                           n_total * 8, kind=2)
    flat_ids_cpu = np.frombuffer(buf, dtype=np.int64).copy()

    # D2H flat_scores
    sbuf = (_ctypes.c_char * (n_total * dtype_size(dt)))()
    DeviceAllocator.memcpy(_ctypes.addressof(sbuf), flat_scores.data_ptr(),
                           n_total * dtype_size(dt), kind=2)

    for dev, global_expert_ids in tables.device_experts.items():
        expert_set = set(global_expert_ids)
        E_local = len(global_expert_ids)

        # Find which flat positions route to experts on this device
        mask = np.isin(flat_ids_cpu, list(expert_set))
        if not mask.any():
            continue

        # original_positions: the flat indices [0..M*top_k) that go to this device
        original_positions = np.where(mask)[0]
        n_local = len(original_positions)

        # Build global expert IDs for moe_align_block_size (NOT remapped)
        # The kernel needs global expert IDs to index the offset table
        local_expert_ids = np.array([flat_ids_cpu[i] for i in original_positions],
                                     dtype=np.int64)

        DeviceAllocator.set_device(dev)
        DeviceAllocator.ensure_triton_device(dev)

        local_expert_ids_gpu = NBXTensor.from_numpy(local_expert_ids)

        # Align tokens for local experts — using GLOBAL expert IDs
        sorted_tids, exp_ids, num_post_pad = \
            moe_align_block_size(local_expert_ids_gpu, _BLOCK_SIZE_M, _num_experts, dev)

        # FIX #1: Remap sorted_tids from local positions back to GLOBAL flat positions
        # sorted_tids contains indices into [0..n_local). We need indices into [0..M*top_k).
        padded_n = sorted_tids.shape[0]
        stid_buf = (_ctypes.c_int64 * padded_n)()
        DeviceAllocator.memcpy(_ctypes.addressof(stid_buf), sorted_tids.data_ptr(),
                               padded_n * 8, kind=2)
        stid_cpu = np.frombuffer(stid_buf, dtype=np.int64).copy()

        # Remap: local index → global flat position
        # Sentinel values (>= n_local) stay as sentinel (use n_total as new sentinel)
        for i in range(padded_n):
            if stid_cpu[i] < n_local:
                stid_cpu[i] = original_positions[stid_cpu[i]]
            else:
                stid_cpu[i] = n_total  # sentinel for kernel (> num_valid_tokens)
        sorted_tids = NBXTensor.from_numpy(stid_cpu)

        # FIX #2: Build local scores matching the global flat positions
        # The kernel indexes flat_scores by sorted_token_ids (now global)
        # So we pass the FULL flat_scores — the kernel will index correctly
        local_scores = _xfer(flat_scores, dev)

        # Transfer activations to this device
        h_local = _xfer(hidden_states, dev)

        # Run 3 GEMM passes + activation
        local_padded = sorted_tids.shape[0]
        gate_out = NBXTensor.zeros((local_padded, N_gate), dtype=dt, device=f"cuda:{dev}")
        w.invoke_fused_moe(
            h_local,
            tables.gate_ptrs[dev],
            gate_out, local_scores,
            sorted_tids, exp_ids, num_post_pad,
            N_gate, K,
            tables.gate_stride_bk[dev], tables.gate_stride_bn[dev],
            top_k, mul_routed_weight=False,
        )

        up_out = NBXTensor.zeros((local_padded, N_gate), dtype=dt, device=f"cuda:{dev}")
        w.invoke_fused_moe(
            h_local,
            tables.up_ptrs[dev],
            up_out, local_scores,
            sorted_tids, exp_ids, num_post_pad,
            N_gate, K,
            tables.up_stride_bk[dev], tables.up_stride_bn[dev],
            top_k, mul_routed_weight=False,
        )

        gate_silu = w.silu(gate_out)
        activated = w.mul(gate_silu, up_out)

        # FIX #3: Down pass writes to [padded, K] with routing weights.
        # Then scatter into output[M, K] using global token positions.
        down_out = NBXTensor.zeros((local_padded, K), dtype=dt, device=f"cuda:{dev}")
        w.invoke_fused_moe(
            activated,
            tables.down_ptrs[dev],
            down_out, local_scores,
            sorted_tids, exp_ids, num_post_pad,
            K, N_gate,
            tables.down_stride_bk[dev], tables.down_stride_bn[dev],
            top_k, mul_routed_weight=True, topk_divide=False,
        )

        # Transfer result back to act_dev
        down_out = _xfer(down_out, act_dev)
        DeviceAllocator.set_device(act_dev)
        DeviceAllocator.ensure_triton_device(act_dev)

        # Combine (device-side, R33-pure). Each row of down_out is the
        # top-k-weighted result for GLOBAL flat position sorted_tids[i] (the
        # down pass already applied mul_routed_weight). Accumulate into
        # output[token_id] where token_id = sorted_tids // top_k. The padding
        # sentinel (sorted_tids == n_total) maps to token_id = M — the sink
        # row of the (M+1, K) output — so no mask is needed. index_add_wrapper
        # preserves down_out's dtype (graph dtype) and accumulates across
        # devices (functional: output_next = output + scatter). No CPU
        # round-trip, no Python token loop, no np.float16 literal — this
        # replaced the pathological host-mediated scatter. (D7 Stage B.)
        stid_dev = _xfer(sorted_tids, act_dev)
        token_ids = w.floor_divide_wrapper(stid_dev, top_k)
        output = w.index_add_wrapper(output, 0, token_ids, down_out)

    # Drop the padding sink row (M); real tokens are [0, M).
    output = output.narrow(0, 0, M).contiguous()
    return output
