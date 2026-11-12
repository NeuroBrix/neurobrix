"""The fused mixture-of-experts layer whose experts are held by several devices: one share per device, summed.

A placement that fills its cards in order can leave one routed layer's experts on two of them. The layer is then
computed where its weights are: for every device of the layer's pointer tables (`memory.pointer_tables.PtrTables`,
whose `device_experts` names the experts each device holds, each device's table kept at the layer's length),

  1. the (token, expert) pairs routed to that device's experts are found on the host, from the routing read back once;
  2. they are aligned by expert on the device, and the aligned ids mapped back to the pairs' positions in the LAYER's
     flat routing — the kernel reads a pair's score and writes its row at that position;
  3. the activations and the scores cross to the device, and the three grouped passes run there (gate and up, SwiGLU,
     down weighted by the routing scores) into outputs of the layer's full row count: this device's pairs filled, the
     others zero;
  4. the device's share of every token — its rows summed over the token's top_k — crosses back and is added on the
     activations' device.

The caller decides nothing here but where the experts are (the tables it built from its placement) and passes plain
values. A layer held by one device does not come here: its alignment stays on the device, with no routing read back.
"""
from __future__ import annotations

import numpy as np

from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.memory.device_transfer import needs_move, transfer_tensor
from nbxtensor.ops import elementwise_wrappers as _elementwise
from nbxtensor.ops import reduction_wrappers as _reduction
from nbxtensor.ops.moe_wrappers import invoke_fused_moe, moe_align_block_size
from nbxtensor.tensor import NBXTensor


def fused_moe_across_devices(hidden_states, flat_indices, flat_scores, tables, top_k: int, num_experts: int,
                             n_gate: int, block_m: int, act_dev: int):
    """The routed SwiGLU sum of a layer whose experts sit on several devices: [M, K] on `act_dev`.

    hidden_states [M, K] and the flat routing (`flat_indices` int64, `flat_scores`, both [M * top_k]) are on
    `act_dev`, the current device; `tables` are the layer's pointer tables (one entry per device in
    `tables.device_experts`); `n_gate` the experts' intermediate width; `block_m` the row block the grouped kernel
    runs at (the alignment's block). The current device is `act_dev` again on return."""
    M, K = hidden_states.shape
    output = NBXTensor.zeros((M, K), dtype=hidden_states.nbx_dtype, device=f"cuda:{act_dev}")
    flat_ids = _int64_on_host(flat_indices)
    for dev, held in tables.device_experts.items():
        mask = np.isin(flat_ids, list(set(held)))
        if not mask.any():
            continue
        positions = np.where(mask)[0]              # the flat positions [0..M*top_k) routed to this device
        _use(dev)
        aligned = _aligned_at_flat_positions(flat_ids, positions, block_m, num_experts, dev, M * top_k)
        down_out = _swiglu_passes(_on(flat_scores, dev), _on(hidden_states, dev), tables, dev, aligned,
                                  (flat_ids.size, n_gate, K, top_k), hidden_states.nbx_dtype)
        # This device's share of every token: its rows summed over the token's top_k, added on the activations' device.
        share = _on(_reduction.sum_wrapper(down_out.reshape(M, top_k, K), dim=1), act_dev)
        _use(act_dev)
        output = _elementwise.add(output, share)
    return output


def _use(dev: int) -> None:
    DeviceAllocator.set_device(dev)
    DeviceAllocator.ensure_triton_device(dev)


def _on(tensor, dev: int):
    """`tensor` on device `dev`: itself when it is there, else a copy through the cross-device transfer (which waits
    for the source device's stream before the peer copy and materialises a view that is not one dense span)."""
    return transfer_tensor(tensor, dev) if needs_move(tensor, dev) else tensor


def _int64_on_host(tensor) -> np.ndarray:
    """An int64 device tensor's elements read back to the host, on the current device."""
    out = np.empty(tensor.numel(), dtype=np.int64)
    DeviceAllocator.memcpy(out.ctypes.data, tensor.data_ptr(), out.nbytes, kind=2)
    return out


def _aligned_at_flat_positions(flat_ids, positions, block_m: int, num_experts: int, dev: int, pairs: int):
    """(sorted ids, block experts, padded count) of one device's pairs, the sorted ids naming the pairs' positions in
    the LAYER's flat routing. The experts keep their number in the layer: a device's pointer tables keep the layer's
    length, so the kernel reads them at that number and the launch keeps the layer's key (`pairs`)."""
    n_total, n_local = flat_ids.size, len(positions)
    local_ids = NBXTensor.from_numpy(flat_ids[positions].astype(np.int64))
    sorted_tids, expert_ids, num_post_pad = moe_align_block_size(local_ids, block_m, num_experts, dev, pairs=pairs)
    # sorted_tids indexes [0..n_local): back to the flat positions; the padding becomes the kernel's sentinel
    # (n_total, not a token)
    local = _int64_on_host(sorted_tids)
    flat = np.where(local < n_local, positions[np.minimum(local, n_local - 1)], n_total)
    return NBXTensor.from_numpy(flat.astype(np.int64)), expert_ids, num_post_pad


def _swiglu_passes(scores, hidden, tables, dev: int, aligned, dims, dtype):
    """The three grouped passes on one device. The kernel writes a pair's row at its flat position (the sorted id it
    reads) and indexes the scores by it, so the FULL scores are passed and every pass's output holds the layer's
    `rows` — this device's pairs filled, the others 0. Returns the down pass's [rows, K], weighted by the scores."""
    rows, n_gate, K, top_k = dims
    gate_out = NBXTensor.zeros((rows, n_gate), dtype=dtype, device=f"cuda:{dev}")
    invoke_fused_moe(hidden, tables.gate_ptrs[dev], gate_out, scores, *aligned, n_gate, K,
                     tables.gate_stride_bk[dev], tables.gate_stride_bn[dev], top_k, mul_routed_weight=False)
    up_out = NBXTensor.zeros((rows, n_gate), dtype=dtype, device=f"cuda:{dev}")
    invoke_fused_moe(hidden, tables.up_ptrs[dev], up_out, scores, *aligned, n_gate, K,
                     tables.up_stride_bk[dev], tables.up_stride_bn[dev], top_k, mul_routed_weight=False)
    activated = _elementwise.mul(_elementwise.silu(gate_out), up_out)
    down_out = NBXTensor.zeros((rows, K), dtype=dtype, device=f"cuda:{dev}")
    invoke_fused_moe(activated, tables.down_ptrs[dev], down_out, scores, *aligned, K, n_gate,
                     tables.down_stride_bk[dev], tables.down_stride_bn[dev], top_k,
                     mul_routed_weight=True, topk_divide=False)
    return down_out
