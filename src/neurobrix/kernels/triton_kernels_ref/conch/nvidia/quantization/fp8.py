# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""FP8 quantization."""

import torch

from conch.kernels.quantization.fp8 import static_scaled_fp8_quant_launcher
from conch.platforms import current_platform


def static_scaled_fp8_quant(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    """Quantize the input tensor to fp8 and return the quantized tensor and scale.

    Args:
        output_tensor: Tensor to write the output of the scaling, shape: (num_tokens, hidden_size).
        input_tensor: Tensor with input to scale, shape: (num_tokens, hidden_size).
        scale: Tensor with static scaling factor to apply, shape: (1).
    """
    assert output_tensor.shape == input_tensor.shape  # noqa: S101
    assert scale.numel() == 1  # noqa: S101

    expected_output_dtype = torch.float8_e4m3fnuz if current_platform.is_amd() else torch.float8_e4m3fn
    assert output_tensor.dtype == expected_output_dtype  # noqa: S101

    static_scaled_fp8_quant_launcher(output_tensor, input_tensor, scale)


def scaled_fp8_quant(
    input_tensor: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled fp8 quantization (static or dynamic).

    Args:
        input_tensor: Tensor to quantize to fp8, shape: (num_tokens, hidden_size).
        scale: (Optional) Scaling factor for quantization. If none, use dynamic, per-token quantization.

    Returns:
        Scaled output tensor and scales.
    """
    if scale is None:
        error_msg = "Dynamic quantization not implemented yet"
        raise NotImplementedError(error_msg)

    # FP8 types described in detail here: https://onnx.ai/onnx/technical/float8.html
    # float8_e4m3fn means 4 bits for exponent (e4), 3 bits for mantissa (m3), no infinity values (fn)
    # ROCm only supports fp8_e4m3fnuz https://rocm.docs.amd.com/en/latest/reference/precision-support.html
    # which means there is no representation of negative zero (uz)
    output_dtype = torch.float8_e4m3fnuz if current_platform.is_amd() else torch.float8_e4m3fn
    output_tensor = torch.zeros_like(input_tensor, dtype=output_dtype)

    static_scaled_fp8_quant(output_tensor, input_tensor, scale)
    return output_tensor, scale
