"""Floor — pure @triton.jit kernel."""

import triton
import triton.language as tl


@triton.jit
def _round_half_even(x):
    """Round to nearest, ties to even — what torch.round and libm nearbyint do.

    Built from `tl.floor` because Triton 3.8.0 exposes no portable `round`,
    `rint` or `nearbyint`; the only implementation available was
    `tl.extra.CUDA.libdevice.nearbyint`, which is NVIDIA's device library and
    fails to lower on every other backend ("PassManager::run failed" on
    triton-ext, measured 2026-09-17).
    """
    r = tl.floor(x + 0.5)
    tie = (r - x) == 0.5
    odd = (r - tl.floor(r * 0.5) * 2.0) != 0.0
    out = tl.where(tie & odd, r - 1.0, r)
    # A zero result keeps the SIGN of its input: round(-0.5) is -0.0, and so are
    # round(-0.3) and round(-0.0). `floor(x + 0.5)` cannot produce -0.0 at all —
    # it maps every x in (-0.5, 0] to +0.0 — so the sign is restored here. The
    # Dell measured this on CUDA against an fp64 oracle: round(-0.5) gave +0.0
    # where main and the oracle give -0.0. The ties were already right; only the
    # sign was lost, and a sign of zero is not cosmetic — it is what tells a
    # downstream divide which infinity to produce.
    #
    # Read from the sign bit rather than from `x < 0`, which is false for -0.0:
    # a bitcast to int32 is negative exactly when the sign bit is set.
    # Restored through the INTEGER view, not a floating select: on CUDA the
    # Dell measured `tl.where(zero & neg, -0.0, out)` returning +0.0 for -0.5,
    # -0.0 and -1e-7 (2026-09-20, against main and the fp64 oracle). Triton
    # compiles floating arithmetic with no-signed-zeros permitted, so a -0.0
    # constant selected into a zero is a value the compiler may write as +0.0.
    # The sign bit copied as a bit is not arithmetic and is never folded.
    xb = x.to(tl.int32, bitcast=True)
    sign_bits = (xb >> 31) << 31                      # 0x80000000 when x is negative, else 0
    ob = out.to(tl.int32, bitcast=True)
    ob = tl.where(out == 0.0, ob | sign_bits, ob)
    return ob.to(tl.float32, bitcast=True)


# Triton's floor / ceil / trunc take fp32 or fp64 only ("Expected dtype
# ['fp32', 'fp64'] but got fp16" — parakeet --triton, 2026-09-05). A half
# input is rounded in fp32 — exact: every half value is an fp32 value and
# the integer result is representable — and stored back in its own dtype;
# fp32 inputs take the same path unchanged.
@triton.jit
def floor_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements
    x = tl.load(input_ptr + offset, mask=mask)
    tl.store(output_ptr + offset, tl.math.floor(x.to(tl.float32)).to(x.dtype), mask=mask)


@triton.jit
def ceil_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements
    x = tl.load(input_ptr + offset, mask=mask)
    tl.store(output_ptr + offset, tl.math.ceil(x.to(tl.float32)).to(x.dtype), mask=mask)


@triton.jit
def round_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements
    x = tl.load(input_ptr + offset, mask=mask)
    # Round-half-to-even, matching torch.round. `libdevice.nearbyint` stood here
    # and is NVIDIA-only: Kokoro-82M died at aten.round::0 (1, 55) with
    # "PassManager::run failed" on triton-ext. A half input is rounded in fp32
    # (exact) and stored back in its dtype (Kokoro --triton, 2026-09-05).
    tl.store(output_ptr + offset,
             _round_half_even(x.to(tl.float32)).to(x.dtype), mask=mask)


@triton.jit
def trunc_forward_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements
    x = tl.load(input_ptr + offset, mask=mask)
    # No trunc in Triton's math: toward zero = floor of the positives, ceil of
    # the negatives (a kernel that could never compile before the bank reached it).
    x32 = x.to(tl.float32)
    tl.store(output_ptr + offset, tl.where(x32 >= 0, tl.math.floor(x32), tl.math.ceil(x32)).to(x.dtype), mask=mask)
