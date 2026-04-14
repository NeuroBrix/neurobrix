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
from collections import defaultdict

import numpy as np

from neurobrix.kernels import wrappers as w
from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator, dtype_size

# Block size for token alignment (must match wrapper's _MOE_BM)
_BLOCK_SIZE_M = 16



# ============================================================================
# POINTER TABLE BUILDER — zero-copy, cached
# ============================================================================

_ptr_cache = {}


class PtrTables:
    """Per-projection absolute pointer tables for zero-copy expert access.

    Each table is [E] int64 on GPU: absolute data_ptr() of each expert.
    The fused kernel loads ptrs[expert_id] and uses it directly as a pointer.
    """
    __slots__ = ('gate_ptrs', 'gate_stride_bk', 'gate_stride_bn',
                 'up_ptrs', 'up_stride_bk', 'up_stride_bn',
                 'down_ptrs', 'down_stride_bk', 'down_stride_bn',
                 'device_experts')

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


def _build_ptr_tables(gate_weights, up_weights, down_weights):
    """Build per-projection absolute pointer tables.

    Stores each expert's data_ptr() as int64 in a GPU tensor. No offset math.
    Total GPU allocation: 3 × E × 8 bytes per device (~3KB for 128 experts).
    """
    num_experts = len(gate_weights)
    tables = PtrTables()
    by_device = defaultdict(list)
    for i in range(num_experts):
        by_device[gate_weights[i]._device_idx].append(i)
    tables.device_experts = dict(by_device)

    for dev, expert_ids in by_device.items():
        DeviceAllocator.set_device(dev)

        def _build_proj(weights):
            # Absolute pointers as int64
            ptrs = np.array(
                [weights[eid].data_ptr() for eid in expert_ids],
                dtype=np.int64)
            ptr_table = NBXTensor.from_numpy(ptrs)
            base = weights[expert_ids[0]]
            return ptr_table, base.stride(1), base.stride(0)

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
# MOE ALIGN BLOCK SIZE — sort tokens by expert on CPU (tiny for decode)
# ============================================================================

def moe_align_block_size(topk_ids_flat, block_size, num_experts, device_idx):
    """Sort tokens by expert, pad to block_size alignment.

    Args:
        topk_ids_flat: NBXTensor [M * top_k] — flat expert indices
        block_size: BLOCK_SIZE_M for the fused kernel
        num_experts: total number of experts
        device_idx: GPU device for output tensors

    Returns:
        sorted_token_ids: NBXTensor [total_padded] — token indices sorted by expert
        expert_ids: NBXTensor [num_blocks] — expert id per block
        num_tokens_post_padded: NBXTensor [1] — total entries after padding
    """
    n = topk_ids_flat.numel()

    # D2H: small tensor (e.g., 8 ints for decode batch=1 top_k=8)
    buf = (_ctypes.c_int64 * n)()
    DeviceAllocator.memcpy(_ctypes.addressof(buf), topk_ids_flat.data_ptr(),
                           n * 8, kind=2)
    ids = np.frombuffer(buf, dtype=np.int64).copy()

    # Count tokens per expert
    counts = np.bincount(ids.clip(0, num_experts - 1), minlength=num_experts)

    # Pad each count to be divisible by block_size
    padded_counts = ((counts + block_size - 1) // block_size) * block_size

    # Cumulative offsets
    offsets = np.zeros(num_experts + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(padded_counts)
    total = int(offsets[-1])

    # Scatter tokens to sorted positions (padding = sentinel n)
    sorted_ids = np.full(total, n, dtype=np.int64)
    cursors = offsets[:-1].copy()
    for i in range(n):
        expert = int(ids[i])
        if 0 <= expert < num_experts:
            sorted_ids[int(cursors[expert])] = i
            cursors[expert] += 1

    # Expert ID per block
    num_blocks = total // block_size
    exp_ids = np.full(num_blocks, -1, dtype=np.int64)
    for e in range(num_experts):
        start_block = int(offsets[e]) // block_size
        end_block = int(offsets[e + 1]) // block_size
        exp_ids[start_block:end_block] = e

    # H2D
    DeviceAllocator.set_device(device_idx)
    sorted_ids_gpu = NBXTensor.from_numpy(sorted_ids)
    expert_ids_gpu = NBXTensor.from_numpy(exp_ids)
    num_post_pad_gpu = NBXTensor.from_numpy(np.array([total], dtype=np.int64))

    return sorted_ids_gpu, expert_ids_gpu, num_post_pad_gpu


# ============================================================================
# D2D TRANSFER HELPER
# ============================================================================

def _xfer(tensor: NBXTensor, target_dev: int) -> NBXTensor:
    """Transfer NBXTensor to target device via D2D memcpy."""
    if tensor._device_idx == target_dev:
        return tensor
    DeviceAllocator.set_device(target_dev)
    dst = NBXTensor.empty(tensor._shape, tensor._dtype, f"cuda:{target_dev}")
    if tensor._nbytes > 0:
        DeviceAllocator.memcpy(dst.data_ptr(), tensor.data_ptr(),
                               tensor._nbytes, kind=3)
    return dst


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
):
    """Execute MoE via fused grouped GEMM — zero torch, zero extra memory.

    API compatible with old per-expert loop. Internally uses fused Triton
    kernels with zero-copy offset tables for expert weight access.

    Args:
        gate_scores: Router logits [batch*seq, num_experts]
        hidden_states: Input activations [batch*seq, hidden_dim]
        gate_weights: List of gate weight NBXTensors per expert
        up_weights: List of up weight NBXTensors per expert
        down_weights: List of down weight NBXTensors per expert
        top_k: Number of experts per token
        num_experts: Total number of experts
        norm_topk_prob: Whether to normalize routing probabilities
        cache_key: Stable key for offset table caching (component + op_uid)
    """
    if hidden_states is None:
        raise RuntimeError("MoE fused: hidden_states is None")

    # Set device context to activation device
    act_dev = hidden_states._device_idx
    DeviceAllocator.set_device(act_dev)
    DeviceAllocator.ensure_triton_device(act_dev)

    # Transfer gate_scores to activation device if needed
    if gate_scores._device_idx != act_dev:
        gate_scores = _xfer(gate_scores, act_dev)

    orig_shape = hidden_states.shape
    if hidden_states.ndim == 3:
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    if gate_scores.ndim == 3:
        gate_scores = gate_scores.reshape(-1, gate_scores.shape[-1])

    # Resolve weight dtype
    w_dtype = gate_weights[0]._dtype
    if hidden_states._dtype != w_dtype:
        hidden_states = hidden_states.to(w_dtype)

    M, K = hidden_states.shape
    N_gate = gate_weights[0].shape[0]  # intermediate_dim (gate/up are [intermediate, hidden])

    # === TEMP DIAG: dump MoE intermediates ===
    import os as _os_moe
    _moe_diag = _os_moe.environ.get("NBX_MOE_DIAG")
    _moe_diag_cache_key = cache_key
    def _dump(label, tensor):
        if not _moe_diag:
            return
        # Only dump for block.1 (first target)
        if "block.1" not in _moe_diag_cache_key:
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
    _dump("gate_scores_in", gate_scores)
    _dump("hidden_states_in", hidden_states)
    # =========================================

    # ================================================================
    # STEP 1: Routing — topk + normalize
    # ================================================================
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
    ck = cache_key if cache_key else f"moe_{gate_weights[0].data_ptr()}_{num_experts}"
    if ck not in _ptr_cache:
        _ptr_cache[ck] = _build_ptr_tables(
            gate_weights, up_weights, down_weights)
        DeviceAllocator.set_device(act_dev)
        DeviceAllocator.ensure_triton_device(act_dev)

    tables = _ptr_cache[ck]

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

    pass

    DeviceAllocator.set_device(act_dev)
    DeviceAllocator.ensure_triton_device(act_dev)

    return output


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

    gate_out = NBXTensor.zeros((padded, N_gate), dtype=dt, device=f"cuda:{dev}")
    w.invoke_fused_moe(
        hidden_states,
        tables.gate_ptrs[dev],
        gate_out, flat_scores,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        N_gate, K,
        tables.gate_stride_bk[dev], tables.gate_stride_bn[dev],
        top_k, mul_routed_weight=False,
    )
    if _diag_dump is not None:
        _diag_dump("gate_out", gate_out)

    up_out = NBXTensor.zeros((padded, N_gate), dtype=dt, device=f"cuda:{dev}")
    w.invoke_fused_moe(
        hidden_states,
        tables.up_ptrs[dev],
        up_out, flat_scores,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        N_gate, K,
        tables.up_stride_bk[dev], tables.up_stride_bn[dev],
        top_k, mul_routed_weight=False,
    )
    if _diag_dump is not None:
        _diag_dump("up_out", up_out)

    # SwiGLU: silu(gate) * up
    gate_silu = w.silu(gate_out)
    activated = w.mul(gate_silu, up_out)
    if _diag_dump is not None:
        _diag_dump("activated", activated)

    # Down pass
    down_out = NBXTensor.zeros((padded, K), dtype=dt, device=f"cuda:{dev}")
    w.invoke_fused_moe(
        activated,
        tables.down_ptrs[dev],
        down_out, flat_scores,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        K, N_gate,
        tables.down_stride_bk[dev], tables.down_stride_bn[dev],
        top_k, mul_routed_weight=True, topk_divide=False,
    )
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
    output = NBXTensor.zeros((M, K), dtype=dt, device=f"cuda:{act_dev}")

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

        # Scatter: down_out is indexed by sorted_tids (global flat positions).
        # The kernel wrote results at positions sorted_tids[i] in down_out.
        # We need to accumulate into output[token_id, :] where token_id = flat_pos // top_k.
        # Read down_out for each valid sorted position and add to output.
        down_buf_bytes = local_padded * K * dtype_size(dt)
        down_cpu_buf = (_ctypes.c_char * down_buf_bytes)()
        DeviceAllocator.memcpy(_ctypes.addressof(down_cpu_buf), down_out.data_ptr(),
                               down_buf_bytes, kind=2)
        down_cpu = np.frombuffer(bytes(down_cpu_buf), dtype=np.float16).reshape(local_padded, K)

        out_bytes = M * K * dtype_size(dt)
        out_cpu_buf = (_ctypes.c_char * out_bytes)()
        DeviceAllocator.memcpy(_ctypes.addressof(out_cpu_buf), output.data_ptr(),
                               out_bytes, kind=2)
        out_cpu = np.frombuffer(bytes(out_cpu_buf), dtype=np.float16).reshape(M, K).copy()

        # The kernel wrote weighted results at sorted positions in down_out.
        # down_out[i] has the result for the token at sorted_tids_cpu[i] // top_k
        for i in range(local_padded):
            flat_pos = int(stid_cpu[i])
            if flat_pos >= n_total:
                continue  # padding sentinel
            token_id = flat_pos // top_k
            out_cpu[token_id] += down_cpu[i].astype(np.float32)

        # Upload accumulated output back
        out_cpu_fp16 = out_cpu.astype(np.float16)
        DeviceAllocator.memcpy(output.data_ptr(),
                               np.ascontiguousarray(out_cpu_fp16).ctypes.data,
                               out_bytes, kind=1)

    return output
