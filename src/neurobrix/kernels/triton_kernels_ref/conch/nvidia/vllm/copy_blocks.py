# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""vLLM copy_blocks."""

from typing import Final

import torch

from conch.kernels.vllm.copy_blocks import copy_blocks_launcher


def _validate_sizes(
    key_caches: list[torch.Tensor],
    value_caches: list[torch.Tensor],
    block_mapping: torch.Tensor,
) -> None:
    """Helper function to validate sizes of input tensors to copy_blocks_launcher.

    Args:
        key_caches: List of key_cache for a set of layers, len: num_layers, each element shape: (num_blocks, cache_block_size * num_kv_heads * head_size).
        value_caches: List of value_cache for a set of layers, len: num_layers, num_layers, each element shape: (num_blocks, cache_block_size * num_kv_heads * head_size).
        block_mapping: List of source/destination pairs, shape: (num_pairs, 2).
    """
    num_layers: Final = len(key_caches)
    if (num_layers_value_cache := len(value_caches)) != num_layers:
        msg = f"Mismatch in number of layers between key_caches ({num_layers}) and value_caches ({num_layers_value_cache})"
        raise ValueError(msg)

    if num_layers == 0:
        msg = "Empty list of kv caches passed to copy_blocks"
        raise ValueError(msg)

    expected_shape: Final = key_caches[0].shape
    expected_kv_cache_dims: Final = 2
    if (actual_len := len(expected_shape)) != expected_kv_cache_dims:
        msg = f"Entry in key cache different-dimensional shape ({actual_len}) than expected ({expected_kv_cache_dims}; shape=(num_cache_blocks, cache_block_size * num_kv_heads * head_size))"
        raise ValueError(msg)

    if any(key_cache.shape != expected_shape for key_cache in key_caches) or any(
        value_cache.shape != expected_shape for value_cache in value_caches
    ):
        msg = "Mismatch in shape of entries in key/value caches"
        raise ValueError(msg)

    expected_stride_0: Final = key_caches[0].stride(0)
    if any(key_cache.stride(0) != expected_stride_0 for key_cache in key_caches) or any(
        value_cache.stride(0) != expected_stride_0 for value_cache in value_caches
    ):
        msg = "Mismatch in stride(0) of entries in key/value caches"
        raise ValueError(msg)

    expected_dtype: Final = key_caches[0].dtype
    if any(key_cache.dtype != expected_dtype for key_cache in key_caches) or any(
        value_cache.dtype != expected_dtype for value_cache in value_caches
    ):
        msg = "Mismatch in dtype of entries in key/value caches"
        raise ValueError(msg)

    expected_block_mapping_dim: Final = 2
    if (block_mapping_dim := len(block_mapping.shape)) != expected_block_mapping_dim:
        msg = f"Block mapping tensor is different-dimensional shape ({block_mapping_dim}) than expected ({expected_block_mapping_dim}; shape=(num_pairs, 2))"
        raise ValueError(msg)

    expected_block_mapping_pair_size: Final = 2
    if block_mapping.size(1) != expected_block_mapping_pair_size:
        msg = f"Block mapping tensor has invalid shape ({block_mapping.shape}), expected shape=(num_pairs, 2))"
        raise ValueError(msg)


def copy_blocks(
    key_caches: list[torch.Tensor],
    value_caches: list[torch.Tensor],
    block_mapping: torch.Tensor,
) -> None:
    """Copy cache blocks from one section of the KV cache to another.

    Args:
        key_caches: List of key_cache for a set of layers, len: num_layers, each element shape: (num_blocks, cache_block_size * num_kv_heads * head_size).
        value_caches: List of value_cache for a set of layers, len: num_layers, num_layers, each element shape: (num_blocks, cache_block_size * num_kv_heads * head_size).
        block_mapping: Tensor in form of list of source/destination pairs, shape: (num_pairs, 2).
    """
    # Verify input sizes/tensor shapes
    _validate_sizes(key_caches, value_caches, block_mapping)

    # Call kernel launch wrapper
    copy_blocks_launcher(key_caches, value_caches, block_mapping)
