# core/validators/tensor_validator.py
"""
Validates tensor shapes at component boundaries.
Catches 5D/4D mismatches before they cause cryptic errors.

ZERO FALLBACK: Invalid shapes crash immediately with helpful message.

Usage:
    from neurobrix.core.validators import TensorValidator

    # Validate latent is 4D before VAE
    TensorValidator.validate_latent_shape(
        tensor=latent,
        component_name="pre_vae",
        expected_dims=4
    )

    # Validate shape matches pattern
    TensorValidator.validate_shape_matches(
        tensor=hidden_states,
        expected_shape=[None, 32, None, None],  # None = any value
        component_name="transformer",
        tensor_name="hidden_states"
    )
"""

import torch
from typing import List, Optional, Tuple


class TensorValidator:
    """
    Validates tensor shapes at component boundaries.

    UNIVERSAL RUNTIME: Works for ALL models, catches common shape errors.
    ZERO FALLBACK: Invalid shapes crash with helpful message.
    """

    @staticmethod
    def validate_latent_shape(
        tensor: torch.Tensor,
        component_name: str,
        expected_dims: int = 4
    ) -> None:
        """
        Validate latent tensor dimensionality.

        ZERO FALLBACK: Crash immediately if wrong dimensionality.

        Args:
            tensor: Tensor to validate
            component_name: For error message
            expected_dims: Expected number of dimensions (default 4 for [B,C,H,W])

        Raises:
            RuntimeError: If tensor has wrong number of dimensions
        """
        if tensor.dim() != expected_dims:
            raise RuntimeError(
                f"\n{'='*70}\n"
                f"ZERO FALLBACK: Shape Validation Failed\n"
                f"{'='*70}\n"
                f"Component: {component_name}\n"
                f"Expected: {expected_dims}D tensor\n"
                f"Got: {tensor.dim()}D tensor with shape {list(tensor.shape)}\n"
                f"\n"
                f"Expected shape pattern: [B, C, H, W] for latents\n"
                f"\n"
                f"Possible causes:\n"
                f"  - Scheduler step returned wrong shape\n"
                f"  - Transformer output not properly reshaped\n"
                f"  - Variable resolver state update issue\n"
                f"{'='*70}"
            )

    @staticmethod
    def validate_shape_matches(
        tensor: torch.Tensor,
        expected_shape: List[Optional[int]],
        component_name: str,
        tensor_name: str
    ) -> None:
        """
        Validate tensor shape matches expected pattern.

        Use None in expected_shape for dimensions that can vary.
        Example: [None, 32, None, None] means batch and spatial dims can vary,
                 but channel dim must be 32.

        Args:
            tensor: Tensor to validate
            expected_shape: Expected shape pattern (None for any)
            component_name: For error message
            tensor_name: Name of tensor for error message

        Raises:
            RuntimeError: If shape doesn't match pattern
        """
        if tensor.dim() != len(expected_shape):
            raise RuntimeError(
                f"\n{'='*70}\n"
                f"ZERO FALLBACK: Shape Dimension Mismatch\n"
                f"{'='*70}\n"
                f"Component: {component_name}\n"
                f"Tensor: {tensor_name}\n"
                f"Expected: {len(expected_shape)} dimensions\n"
                f"Got: {tensor.dim()} dimensions\n"
                f"Shape: {list(tensor.shape)}\n"
                f"Pattern: {expected_shape}\n"
                f"{'='*70}"
            )

        mismatches = []
        for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
            if expected is not None and actual != expected:
                mismatches.append((i, actual, expected))

        if mismatches:
            mismatch_str = "\n".join(
                f"  - Dim {i}: got {actual}, expected {expected}"
                for i, actual, expected in mismatches
            )
            raise RuntimeError(
                f"\n{'='*70}\n"
                f"ZERO FALLBACK: Shape Value Mismatch\n"
                f"{'='*70}\n"
                f"Component: {component_name}\n"
                f"Tensor: {tensor_name}\n"
                f"Shape: {list(tensor.shape)}\n"
                f"Pattern: {expected_shape}\n"
                f"\n"
                f"Mismatches:\n{mismatch_str}\n"
                f"{'='*70}"
            )


