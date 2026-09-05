"""A trivial kernel in a real file — Triton needs source it can read."""

import triton
import triton.language as tl


@triton.jit
def touch(p, BLOCK: tl.constexpr):
    tl.store(p + tl.arange(0, BLOCK), 1.0)
