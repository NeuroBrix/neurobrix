# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""GeLU Tanh and Mul."""

import torch

from conch.kernels.activation.gelu_tanh_and_mul import gelu_tanh_and_mul_launcher


def gelu_tanh_and_mul(x: torch.Tensor) -> torch.Tensor:
    """gelu_tanh_and_mul operation."""
    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)

    gelu_tanh_and_mul_launcher(output, x)

    return output
