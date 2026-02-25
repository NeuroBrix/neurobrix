"""Quantization primitives implemented in triton."""


import triton
import triton.language as tl


@triton.jit
def clamp(x: tl.tensor, min_val, max_val) -> tl.tensor:
    """Clamps all elements in `x` into range [min, max].

    Args:
        x (tl.tensor): the input tensor.
        min_val (Number): lower bound of the range.
        max_val (Number): upper bound of the range.

    Returns:
        tl.tensor: the output tensor.
    """
    return tl.math.min(tl.math.max(x, min_val), max_val)


@triton.jit
def dequantize(x: tl.tensor, scale: tl.tensor) -> tl.tensor:
    """Dequantize quantized tensor to floating point.

    Args:
        x (tl.tensor): quantized tensor.
        scale (tl.tensor): quantization scaling factor

    Returns:
        tl.tensor: Dequantized floating-point tensor.
    """
    return (x * scale).to(tl.float32)


@triton.jit
def quantize(x, scale, qmin, qmax) -> tl.tensor:
    """Quantize the tensor given quantization scale and data type.

    Args:
        x (tl.tensor): floating-point tensor
        scale (tl.tensor): quantization scale factor.
        qmin (Number): quantization minimum range.
        qmax (Number): quantization maximum range

    Returns:
        tl.tensor: rounded and clamped tensor.
            Note: this is still in floating point as we can't pass dtype to function

    Example:
    
        out = quantize(out, scale, -128, 127).to(tl.int8)
    """
    return clamp(tl.math.round(x / scale), qmin, qmax)
