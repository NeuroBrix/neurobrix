"""Random number generation — pure @triton.jit kernels.

rand_kernel: uniform [0, 1) using tl.rand(seed, offset)
randn_kernel: standard normal N(0,1) using tl.randn(seed, offset)

Both use Triton's built-in PRNG which implements Philox counter-based RNG.
"""

import triton
import triton.language as tl

@triton.jit
def rand_kernel(
    output_ptr,
    n_elements,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    """Fill output with uniform random values in [0, 1).

    Args:
        output_ptr: pointer to output tensor (flattened)
        n_elements: total number of elements
        seed: RNG seed (int)
    """
    pid = tl.program_id(0)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements

    r = tl.rand(seed, offset)
    tl.store(output_ptr + offset, r, mask=mask)

@triton.jit
def randn_kernel(
    output_ptr,
    n_elements,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    """Fill output with standard normal random values N(0, 1).

    Args:
        output_ptr: pointer to output tensor (flattened)
        n_elements: total number of elements
        seed: RNG seed (int)
    """
    pid = tl.program_id(0)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements

    r = tl.randn(seed, offset)
    tl.store(output_ptr + offset, r, mask=mask)
