"""Remainder (modulo) — pure @triton.jit kernel.

torch.remainder / Python `%` define the result to follow the sign of the
DIVISOR: remainder(a, b) = a - floor(a / b) * b, so the result lies in [0, b)
for b > 0 (e.g. remainder(-0.1, 1) = 0.9). Triton's `%` operator on floats is C
`fmod` (trunc division), which follows the sign of the DIVIDEND (fmod(-0.1, 1) =
-0.1). Using `%` silently broke any negative-input modulo — surfaced by the
Kokoro SineGen phase wrap `(f0*harmonics/sr) % 1`, where the predictor's
unconstrained (sometimes negative) F0 produced negative phases that never
wrapped into [0,1), degrading the harmonic source. The floor formula matches
torch for floats and for integers (the result is integer-valued, cast back
cleanly); x/y promotes to float so `floor` is well-defined for int inputs too.
"""

import triton
import triton.language as tl

@triton.jit
def remainder_forward_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)
    result = x - tl.math.floor(x / y) * y
    tl.store(output_ptr + offset, result, mask=mask)

@triton.jit
def remainder_scalar_kernel(
    x_ptr, output_ptr,
    n_elements,
    divisor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask)
    result = x - tl.math.floor(x / divisor) * divisor
    tl.store(output_ptr + offset, result, mask=mask)
