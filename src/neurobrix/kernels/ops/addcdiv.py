"""Addcdiv kernel — extracted from FlagGems.

Pure Triton. ZERO PyTorch imports.
Logic: result = input + value * (tensor1 / tensor2)
"""

import triton
import triton.language as tl

@triton.jit
def addcdiv_forward_kernel(
    input_ptr,
    tensor1_ptr,
    tensor2_ptr,
    output_ptr,
    n_elements,
    value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(input_ptr + offset, mask=mask)
    t1 = tl.load(tensor1_ptr + offset, mask=mask)
    t2 = tl.load(tensor2_ptr + offset, mask=mask)

    result = x + value * (t1 / t2)
    tl.store(output_ptr + offset, result, mask=mask)
