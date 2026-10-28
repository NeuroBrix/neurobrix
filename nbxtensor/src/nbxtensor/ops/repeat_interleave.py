"""repeat_interleave — pure @triton.jit kernel.

Handles the tensor-repeats case: given a 1D repeats tensor and its cumsum,
writes indices so that element i is repeated repeats[i] times.

Example: repeats=[2,3,1], cumsum=[2,5,6] -> output=[0,0,1,1,1,2]

The self-int case (scalar repeats) is a stride trick handled in the wrapper.

Ported from FlagGems repeat_interleave.py. Stripped FlagGems decorators.
"""

import triton
import triton.language as tl


@triton.jit
def repeat_interleave_tensor_kernel(
    repeats_ptr,
    cumsum_ptr,
    out_ptr,
    size,
    BLOCK_SIZE: tl.constexpr,
):
    """Write source indices for tensor-based repeat_interleave.

    Grid: (size,) — one program per input element.
    Each program writes its index `pid` into repeats[pid] consecutive
    positions starting at cumsum[pid] - repeats[pid].
    """
    pid = tl.program_id(0).to(tl.int64)
    mask = pid < size
    cumsum = tl.load(cumsum_ptr + pid, mask, other=0)
    # The loop bound below must be 32-bit: the Metal induction lowering does
    # not terminate on a 64-bit `scf.for` bound (it refuses rather than hang).
    # `repeats` arrives i64 by torch default. The cast is safe -- a per-element
    # repeat count of 2**31 would need an 8 GB output for ONE element, which
    # the allocator refuses long before this loop -- and numerically inert on
    # CUDA: same iterations, narrower type. Addressing stays wide: `out_offset`
    # promotes back through `cumsum`.
    repeats = tl.load(repeats_ptr + pid, mask, other=0).to(tl.int32)
    out_offset = cumsum - repeats

    base_ptr = out_ptr + out_offset
    for start_k in range(0, repeats, BLOCK_SIZE):
        offsets_k = start_k + tl.arange(0, BLOCK_SIZE)
        mask_k = offsets_k < repeats
        tl.store(base_ptr + offsets_k, pid, mask=mask_k)
