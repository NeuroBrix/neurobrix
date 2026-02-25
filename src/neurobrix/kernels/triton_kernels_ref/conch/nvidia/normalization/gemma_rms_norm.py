# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Gemma-style RMS Norm."""

import torch

from conch.kernels.normalization.gemma_rms_norm import gemma_rms_norm_inplace_launcher


def gemma_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Gemma RMS norm operation."""
    if residual is not None:
        x = x + residual
        residual = x

    gemma_rms_norm_inplace_launcher(x, weight, variance_epsilon)

    return x if residual is None else (x, residual)
