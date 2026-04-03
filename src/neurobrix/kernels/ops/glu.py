"""GLU (Gated Linear Unit) — pure @triton.jit kernel. Ported from attorch (MIT).

Supports arbitrary activation functions via the gate path.
SwiGLU = GLU with SiLU activation.
"""

import triton
import triton.language as tl

from ._common import sigmoid

@triton.jit
def glu_forward_kernel(
    input1_ptr, input2_ptr, output_ptr,
    size,
    act_func: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """GLU forward: out = input1 * act(input2).

    input1, input2: [size] (the two halves of the gated input)
    act_func: 'sigmoid' (standard GLU), 'silu' (SwiGLU), 'gelu', 'relu'
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    x1 = tl.load(input1_ptr + offset, mask=mask)
    x2 = tl.load(input2_ptr + offset, mask=mask)

    x2_fp32 = x2.to(tl.float32)
    if act_func == 'sigmoid':
        gate = sigmoid(x2_fp32)
    elif act_func == 'silu':
        gate = x2_fp32 * sigmoid(x2_fp32)
    elif act_func == 'gelu':
        gate = 0.5 * x2_fp32 * (1.0 + tl.math.erf(0.707106781 * x2_fp32))
    elif act_func == 'relu':
        gate = tl.maximum(0, x2_fp32)
    else:
        gate = sigmoid(x2_fp32)

    tl.store(output_ptr + offset, x1 * gate, mask=mask)
