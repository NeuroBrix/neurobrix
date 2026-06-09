"""Cross-device NBXTensor transfer — shared by the compiled (TritonSequence)
and op-by-op (triton_sequential) execution paths.

Pipeline-parallel placement shards a component's weights across GPUs (block N
on cuda:i, block N+k on cuda:j). At a stage boundary the activation produced on
the previous stage's device must be moved to the next stage's device before the
op runs — `NBXTensor.to('cuda:N')` is intentionally a no-op (it only casts
dtype), so the move is a real D2D `memcpy` here. R33-pure (DeviceAllocator +
NBXTensor only; no torch). Extracted from `sequence.py:_transfer_tensor` /
`_needs_move` so both modes share ONE implementation (R30 parity by
construction). Model-agnostic.
"""

from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator


def needs_move(t: NBXTensor, target_dev: int) -> bool:
    """True iff `t` must be transferred to land on cuda:target_dev.

    1. CPU source (zero3 offload) → must H2D (even if `_device_idx` is 0).
    2. Different CUDA device → must D2D.
    3. Already on target → no-op.
    """
    if getattr(t, "_device", "cuda") == "cpu":
        return True
    return t._device_idx != target_dev


def transfer_tensor(tensor: NBXTensor, target_dev: int) -> NBXTensor:
    """Copy an NBXTensor to cuda:target_dev, preserving shape/strides/dtype.

    kind=1 (H2D) for a CPU source (zero3), kind=3 (D2D) for a cross-GPU move.
    Expanded/broadcast views (stride 0) are materialised first so the memcpy
    does not over-read the backing allocation. Strides are carried over (not
    forced contiguous) — a pre-transposed weight view keeps its `.t()` stride
    semantics, matching the native `torch.Tensor.to(device)` contract; a
    downstream `.contiguous()` inside mm/bmm materialises correctly.
    """
    if tensor.is_expanded():
        tensor = tensor.contiguous()
    DeviceAllocator.set_device(target_dev)
    src_device = getattr(tensor, "_device", "cuda")
    kind = 1 if src_device == "cpu" else 3
    if tensor._nbytes > 0:
        dst_raw_ptr = DeviceAllocator.malloc_cuda(tensor._nbytes)
        DeviceAllocator.memcpy(dst_raw_ptr, tensor.data_ptr(),
                               tensor._nbytes, kind=kind)
    else:
        dst_raw_ptr = 0
    return NBXTensor(
        dst_raw_ptr, tensor._shape, tensor._strides, tensor._dtype,
        "cuda", owns_data=True, device_idx=target_dev, offset=0)
