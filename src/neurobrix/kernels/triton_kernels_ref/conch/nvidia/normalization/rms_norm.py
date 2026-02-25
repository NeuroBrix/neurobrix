# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""RMS norm."""

import torch

from conch.kernels.normalization.rms_norm import fused_add_rms_norm_launcher, rms_norm_launcher


def rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Root-mean-square normalization.

    Args:
        x: Input tensor, of shape (..., hidden_size).
        weight: Weight tensor, of shape (hidden_size,).
        epsilon: Epsilon value.
    """
    output = torch.empty_like(x, dtype=x.dtype, device=x.device)

    # Call kernel launch wrapper
    rms_norm_launcher(output, x, weight, epsilon)

    return output


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Root-mean-square normalization with fused add.

    Args:
        x: Input tensor, of shape (..., hidden_size).
        residual: Residual tensor, of shape (..., hidden_size).
        weight: Weight tensor, of shape (hidden_size,).
        epsilon: Epsilon value.
    """
    fused_add_rms_norm_launcher(x, residual, weight, epsilon)
    return x, residual
