# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""GEMM kernel supporting mixed-precision (e.g. w{1|2|4|8|16}, a{16|32}) and scaled matrix multiplications (fp8/int8 with scaling factors to different output dtype)."""

from typing import Final

import torch

from conch.kernels.quantization.gemm import (
    ChannelScaleMode,
    MixedPrecisionMatmulMetadata,
    ScaledMatmulMetadata,
    WeightGroupMode,
    mixed_precision_gemm_launcher,
    scaled_gemm_launcher,
)


def _check_contiguous(
    x: torch.Tensor,
    w_q_packed: torch.Tensor,
    w_s: torch.Tensor,
    w_zp: torch.Tensor | None,
) -> bool:
    """Check if tensors are contiguous."""
    return (
        x.is_contiguous() and w_q_packed.is_contiguous() and w_s.is_contiguous() and w_zp.is_contiguous()
        if w_zp is not None
        else True
    )


def _deduce_weight_group_mode(
    w_zp: torch.Tensor | None,
) -> WeightGroupMode:
    """Deduce the weight group mode."""
    return WeightGroupMode.SYMMETRIC_NO_SHIFT if w_zp is None else WeightGroupMode.SYMMETRIC_WITH_SHIFT


def create_mixed_precision_metadata(
    x: torch.Tensor,
    w_q_packed: torch.Tensor,
    w_s: torch.Tensor,
    w_zp: torch.Tensor | None,
    weight_size_bits: int,
    weight_bias: int,
    group_size: int,
    *,
    output_dtype: torch.dtype | None = None,
    acc_dtype: torch.dtype | None = None,
    meta_dtype: torch.dtype | None = None,
    scaled_activations: bool = False,
    strict: bool = False,
) -> MixedPrecisionMatmulMetadata:
    """Verify sizes and dtypes of tensors and deduce metadata parameters."""
    if strict:
        expected_input_matrix_rank: Final = 2

        if (x_rank := len(x.shape)) != expected_input_matrix_rank:
            error_msg = f"Unexpected number of dimensions of input tensor x: {x_rank}"
            raise ValueError(error_msg)

        if (w_q_packed_rank := len(w_q_packed.shape)) != expected_input_matrix_rank:
            error_msg = f"Unexpected number of dimensions of input tensor w_q_packed: {w_q_packed_rank}"
            raise ValueError(error_msg)

        if (w_s_rank := len(w_s.shape)) != expected_input_matrix_rank:
            error_msg = f"Unexpected number of dimensions of input tensor w_s: {w_s_rank}"
            raise ValueError(error_msg)

        if w_zp is not None and (w_zp_rank := len(w_zp.shape)) != expected_input_matrix_rank:
            error_msg = f"Unexpected number of dimensions of input tensor w_zp: {w_zp_rank}"
            raise ValueError(error_msg)

        # Expecting some form of 32-bit packing
        expected_packed_dtypes: Final = [torch.uint32, torch.int32]
        if (packed_dtype := w_q_packed.dtype) not in expected_packed_dtypes:
            error_msg = f"Invalid datatype for packed weights: {packed_dtype}"
            raise ValueError(error_msg)

    # Assume 32-bit packing
    packed_bitwidth: Final = 32
    elements_per_sample = packed_bitwidth // weight_size_bits

    m_dim, k_dim = x.shape
    _, n_dim = w_q_packed.shape

    unpack_mask = 2**weight_size_bits - 1

    # Check if zeros is a scalar value
    zero_is_scalar = False if w_zp is None else w_zp.numel() == 1

    if strict:
        # Verify shape of w_s
        expected_scales_shape: Final = (k_dim // group_size, n_dim)
        if (scales_shape := w_s.shape) != expected_scales_shape:
            error_msg = f"Invalid w_s shape (expected: {expected_scales_shape}, actual: {scales_shape})"
            raise ValueError(error_msg)

        # Expected shape of zeros tensor if 1) it is not scalar 2) it is not None
        expected_zeros_shape: Final = (k_dim // group_size, n_dim)
        # Verify shape of w_zp
        if not zero_is_scalar and w_zp is not None and (zeros_shape := w_zp.shape) != expected_zeros_shape:
            error_msg = f"Invalid w_zp shape (expected: {expected_zeros_shape}, actual: {zeros_shape})"
            raise ValueError(error_msg)

        # Not supporting scaled activations right now, but we can add support later if needed. This simplifies the interface
        if scaled_activations:
            error_msg = "Scaled activations not yet implemented (need to deduce correct channel_scale_mode)"
            raise NotImplementedError(error_msg)

    return MixedPrecisionMatmulMetadata(
        m_dim=m_dim,
        k_dim=k_dim,
        n_dim=n_dim,
        weight_size_bits=weight_size_bits,
        weight_bias=weight_bias,
        group_size=group_size,
        elements_per_sample=elements_per_sample,
        zero_is_scalar=zero_is_scalar,
        unpack_mask=unpack_mask,
        data_contiguous=_check_contiguous(x, w_q_packed, w_s, w_zp),
        input_dtype=x.dtype,
        output_dtype=x.dtype if output_dtype is None else output_dtype,
        acc_dtype=torch.float32 if acc_dtype is None else acc_dtype,
        meta_dtype=x.dtype if meta_dtype is None else meta_dtype,
        channel_scale_mode=ChannelScaleMode.NONE,
        weight_group_mode=_deduce_weight_group_mode(w_zp),
    )


def mixed_precision_gemm(
    x: torch.Tensor,
    w_q_packed: torch.Tensor,
    w_s: torch.Tensor,
    w_zp: torch.Tensor | None,
    weight_size_bits: int,
    weight_bias: int,
    group_size: int,
    *,
    output_dtype: torch.dtype | None = None,
    acc_dtype: torch.dtype | None = None,
    meta_dtype: torch.dtype | None = None,
    scaled_activations: bool = False,
    strict: bool = False,
) -> torch.Tensor:
    """Mixed precision GEMM operation."""
    metadata = create_mixed_precision_metadata(
        x,
        w_q_packed,
        w_s,
        w_zp,
        weight_size_bits,
        weight_bias,
        group_size,
        output_dtype=output_dtype,
        acc_dtype=acc_dtype,
        meta_dtype=meta_dtype,
        scaled_activations=scaled_activations,
        strict=strict,
    )

    output = torch.zeros((metadata.m_dim, metadata.n_dim), device=x.device, dtype=metadata.output_dtype)

    mixed_precision_gemm_launcher(output, x, w_q_packed, w_s, w_zp, metadata)

    return output


def create_scaled_metadata(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    output_dtype: torch.dtype,
    strict: bool = False,
) -> ScaledMatmulMetadata:
    """Verify sizes and dtypes of tensors and deduce metadata parameters."""
    if strict:
        expected_input_matrix_rank: Final = 2

        if (a_rank := len(a.shape)) != expected_input_matrix_rank:
            error_msg = f"Unexpected number of dimensions of input tensor a: {a_rank}"
            raise ValueError(error_msg)

        if (b_rank := len(b.shape)) != expected_input_matrix_rank:
            error_msg = f"Unexpected number of dimensions of input tensor b: {b_rank}"
            raise ValueError(error_msg)

        if a.dtype != b.dtype:
            error_msg = f"Input tensors a and b must have the same datatype (a: {a.dtype}, b: {b.dtype})"
            raise ValueError(error_msg)

    m_dim, k_dim = a.shape
    _, n_dim = b.shape

    if strict:
        if scale_a.numel() != 1:
            if (scale_a_rank := len(scale_a.shape)) != expected_input_matrix_rank:
                error_msg = f"Unexpected number of dimensions of input tensor scale_a: {scale_a_rank}"
                raise ValueError(error_msg)

            if scale_a.shape[0] != m_dim:
                error_msg = f"Invalid scale_a shape (expected: ({m_dim},), actual: {scale_a.shape})"
                raise ValueError(error_msg)

        if scale_b.numel() != 1:
            if (scale_b_rank := len(scale_b.shape)) != expected_input_matrix_rank:
                error_msg = f"Unexpected number of dimensions of input tensor scale_b: {scale_b_rank}"
                raise ValueError(error_msg)

            if scale_b.shape[0] != n_dim:
                error_msg = f"Invalid scale_b shape (expected: ({n_dim},), actual: {scale_b.shape})"
                raise ValueError(error_msg)

    return ScaledMatmulMetadata(
        m_dim=m_dim,
        k_dim=k_dim,
        n_dim=n_dim,
        data_contiguous=(
            a.is_contiguous() and b.is_contiguous() and scale_a.is_contiguous() and scale_b.is_contiguous()
        ),
        input_dtype=a.dtype,
        output_dtype=output_dtype,
        acc_dtype=torch.float32 if a.is_floating_point() else torch.int32,
        meta_dtype=scale_a.dtype,
        channel_scale_mode=ChannelScaleMode.WEIGHT_AND_ACTIVATION,
        weight_group_mode=WeightGroupMode.NONE,
    )


def scaled_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    output_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
    strict: bool = False,
) -> torch.Tensor:
    """Scaled GEMM operation."""
    metadata = create_scaled_metadata(a, b, scale_a, scale_b, output_dtype, strict=strict)

    output = torch.zeros((metadata.m_dim, metadata.n_dim), device=a.device, dtype=output_dtype)

    scaled_gemm_launcher(output, a, b, scale_a, scale_b, metadata)

    if bias is not None:
        output.add_(bias)

    return output
