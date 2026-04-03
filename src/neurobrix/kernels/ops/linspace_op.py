"""Linspace — pure @triton.jit kernel.

Generate evenly spaced values between start and end.
Uses bidirectional computation for numerical stability:
    forward:  value = start + idx * step
    backward: value = end - (steps - idx - 1) * step
    result:   pick forward for first half, backward for second half.

Extracted from FlagGems (Apache 2.0).

Grid: (ceil(steps / BLOCK_SIZE),)
"""

import triton
import triton.language as tl


@triton.jit
def linspace_kernel(
    out_ptr,
    out_stride0,
    start,
    mid,
    end,
    step_size,
    steps,
    BLOCK_SIZE: tl.constexpr,
):
    """Generate evenly spaced values from start to end.

    Uses bidirectional computation: forward from start for the first half,
    backward from end for the second half. This reduces floating-point
    drift for large step counts — the endpoints are always exact.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < steps

    # Forward: start + step * idx (accurate near start)
    fw_mask = idx < mid
    fw_values = start + (step_size * idx)

    # Backward: end - step * (steps - idx - 1) (accurate near end)
    bd_values = end - step_size * (steps - idx - 1)

    out_val = tl.where(fw_mask, fw_values, bd_values)
    tl.store(out_ptr + idx * out_stride0, out_val, mask=mask)
