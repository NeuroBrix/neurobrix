"""Floor — pure @triton.jit kernel."""

import triton
import triton.language as tl


@triton.jit
def floor_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(input_ptr + offset, mask=mask)
    tl.store(output_ptr + offset, tl.math.floor(x), mask=mask)


@triton.jit
def ceil_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(input_ptr + offset, mask=mask)
    tl.store(output_ptr + offset, tl.math.ceil(x), mask=mask)


@triton.jit
def round_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(input_ptr + offset, mask=mask)
    # tl.math.nearbyint does not exist in Triton 3.6; libdevice.nearbyint is the
    # round-half-to-even primitive matching torch.round (cf. pow.py using
    # tl.extra.cuda.libdevice.pow). The round kernel was crashing before this.
    tl.store(output_ptr + offset, tl.extra.cuda.libdevice.nearbyint(x), mask=mask)


@triton.jit
def trunc_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(input_ptr + offset, mask=mask)
    tl.store(output_ptr + offset, tl.math.trunc(x), mask=mask)
