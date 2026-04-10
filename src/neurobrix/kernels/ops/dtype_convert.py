"""GPU dtype conversion kernels — bf16↔fp16 via bit manipulation.

Eliminates CPU-side numpy conversion during weight loading.
Raw bf16 bytes are H2D copied as uint16, then converted on GPU.
"""

import triton
import triton.language as tl


@triton.jit
def bf16_to_fp16_kernel(
    src_ptr,   # uint16 (bf16 raw bits) on GPU
    dst_ptr,   # fp16 output on GPU
    N,         # total number of elements
    BLOCK: tl.constexpr,
):
    """Convert bf16 → fp16 on GPU via fp32 intermediate.

    bf16 bits:  [sign(1)][exp(8)][mantissa(7)]
    fp32 bits:  [sign(1)][exp(8)][mantissa(23)]  ← bf16 << 16
    fp16 bits:  [sign(1)][exp(5)][mantissa(10)]  ← GPU truncates fp32→fp16

    The GPU handles fp32→fp16 truncation natively (with rounding).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Load bf16 raw bits (stored as int16, same 2-byte layout)
    raw_i16 = tl.load(src_ptr + offs, mask=mask, other=0)
    # Mask to 16 bits to prevent sign extension when widening to uint32
    raw = (raw_i16.to(tl.int32) & 0xFFFF).to(tl.uint32)

    # bf16 → fp32: shift left 16 bits, bitcast to float32
    fp32_bits = raw << 16
    fp32_vals = fp32_bits.to(tl.float32, bitcast=True)

    # fp32 → fp16: GPU truncates natively
    fp16_vals = fp32_vals.to(tl.float16)

    tl.store(dst_ptr + offs, fp16_vals, mask=mask)
