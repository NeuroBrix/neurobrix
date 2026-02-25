# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Int8 quantization."""

import torch

from conch.kernels.quantization.int8 import static_scaled_int8_quant_launcher


def static_scaled_int8_quant(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    """Quantize the input tensor to int8 and return the quantized tensor and scale.

    Args:
        output_tensor: Tensor to write the output of the scaling, shape: (num_tokens, hidden_size).
        input_tensor: Tensor with input to scale, shape: (num_tokens, hidden_size).
        scale: Tensor with static scaling factor to apply, shape: (1).
    """
    assert output_tensor.shape == input_tensor.shape  # noqa: S101
    assert scale.numel() == 1  # noqa: S101

    static_scaled_int8_quant_launcher(output_tensor, input_tensor, scale)


def scaled_int8_quant(
    input_tensor: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled int8 quantization (static or dynamic).

    Args:
        input_tensor: Tensor to quantize to int8, shape: (num_tokens, hidden_size).
        scale: (Optional) Scaling factor for quantization. If none, use dynamic, per-token quantization.

    Returns:
        Scaled output tensor and scales.
    """
    if scale is None:
        error_msg = "Dynamic int8 quantization not yet implemented"
        raise NotImplementedError(error_msg)

    output_tensor = torch.zeros_like(input_tensor, dtype=torch.int8)
    static_scaled_int8_quant(output_tensor, input_tensor, scale)
    return output_tensor, scale
