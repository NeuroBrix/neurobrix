"""Floor — pure @triton.jit kernel."""

import triton
import triton.language as tl


# Triton's floor / ceil / trunc take fp32 or fp64 only ("Expected dtype
# ['fp32', 'fp64'] but got fp16" — parakeet --triton, 2026-09-05). A half
# input is rounded in fp32 — exact: every half value is an fp32 value and
# the integer result is representable — and stored back in its own dtype;
# fp32 inputs take the same path unchanged.
@triton.jit
def floor_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(input_ptr + offset, mask=mask)
    tl.store(output_ptr + offset, tl.math.floor(x.to(tl.float32)).to(x.dtype), mask=mask)


@triton.jit
def ceil_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(input_ptr + offset, mask=mask)
    tl.store(output_ptr + offset, tl.math.ceil(x.to(tl.float32)).to(x.dtype), mask=mask)


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
    # No trunc in Triton's math: toward zero = floor of the positives, ceil of
    # the negatives (a kernel that could never compile before the bank reached it).
    x32 = x.to(tl.float32)
    tl.store(output_ptr + offset, tl.where(x32 >= 0, tl.math.floor(x32), tl.math.ceil(x32)).to(x.dtype), mask=mask)
