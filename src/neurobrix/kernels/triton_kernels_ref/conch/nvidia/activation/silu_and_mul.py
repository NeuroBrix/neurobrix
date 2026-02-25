# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""SiLU and Mul."""

import torch

from conch.kernels.activation.silu_and_mul import silu_and_mul_launcher


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Args:
        x: Input tensor, of shape (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d).

    Returns:
        Output tensor, of shape (num_tokens, d) or (batch_size, seq_len, d).
    """
    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)

    # Call kernel launch wrapper
    silu_and_mul_launcher(output, x)

    return output
