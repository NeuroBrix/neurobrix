"""Shared activation element functions — pure @triton.jit, no pointers.

These operate on already-loaded values and compose naturally.
Ported from attorch (MIT license).
"""

import triton
import triton.language as tl


@triton.jit
def sigmoid(x):
    """Sigmoid: 1 / (1 + exp(-x))"""
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def tanh_fn(x):
    """Tanh via sigmoid: 2 * sigmoid(2x) - 1"""
    return 2.0 * sigmoid(2.0 * x) - 1.0


@triton.jit
def relu_fn(x):
    """ReLU: max(0, x)"""
    return tl.maximum(0, x)


@triton.jit
def relu6_fn(x):
    """ReLU6: min(max(0, x), 6)"""
    return tl.minimum(relu_fn(x), 6)
