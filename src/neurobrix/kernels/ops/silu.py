"""SiLU (Swish) — pure @triton.jit kernel."""

import triton
import triton.language as tl

from ._common import sigmoid

@triton.jit
def silu_forward_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Cast offset to int64 so that mask `offset < n_elements`
    # works correctly when n_elements >= 2^31 (e.g. Sana 4Kpx VAE
    # silu input 1*128*4096*4096 = 2^31 elements). Without this,
    # int32 n_elements wraps to negative and the mask is all-False,
    # silently skipping every element and leaving output as
    # uninitialized memory (= garbage).
    offset = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    mask = offset < n_elements

    x = tl.load(input_ptr + offset, mask=mask)
    x_fp32 = x.to(tl.float32)
    out = x_fp32 * sigmoid(x_fp32)
    tl.store(output_ptr + offset, out, mask=mask)
