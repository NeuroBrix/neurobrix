# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Flash Attention w/ Paged KV Cache + FlashDecoding."""

from dataclasses import dataclass
from typing import Final

import torch

from conch.kernels.attention.paged_attention import paged_attention_launcher


@dataclass
class PagedAttentionMetadata:
    """Wrapper class holding metadata for PagedAttention kernel."""

    batch_size: int
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    num_cache_blocks: int
    max_num_blocks_per_sequence: int
    max_num_splits: int


def _check_output_query_size_compatibility(out: torch.Tensor, query: torch.Tensor) -> None:
    """Check size compatibility of Output and Query tensors.

    Args:
        out: Output tensor.
        query: Query tensor.

    Raises:
        ValueError if sizes are mismatched.
    """
    # Output tensor should be a 3-D tensor of shape (batch_size, num_heads, head_size)
    expected_output_shape_dims: Final = 3

    if len(out_shape := out.shape) != expected_output_shape_dims:
        msg = f"Output tensor has unexpected shape (shape={out_shape}), expected {expected_output_shape_dims}-D tensor"
        raise ValueError(msg)

    # Query tensor should have same shape as output
    if len(query_shape := query.shape) != expected_output_shape_dims:
        msg = f"Query tensor has unexpected shape (shape={query_shape}), expected {expected_output_shape_dims}-D tensor"
        raise ValueError(msg)

    if query_shape != out_shape:
        msg = f"Shape of query and output tensors does not match (query.shape={query_shape}, out.shape={out_shape})"
        raise ValueError(msg)


def _check_kv_cache_size_compatibility(key_cache: torch.Tensor, value_cache: torch.Tensor, head_size: int) -> None:
    """Check size compatibility of KV cache tensor.

    Args:
        key_cache: Key cache tensor.
        value_cache: Key cache tensor.
        head_size: Size of attention head, deduced from out/query tensor sizes.

    Raises:
        ValueError if sizes are mismatched.
    """
    # Key/Value Cache tensor should be a 4-D tensor of shape (num_blocks, cache_block_size, num_kv_heads, head_size)
    expected_kv_cache_shape_dims: Final = 4

    if len(key_cache.shape) != expected_kv_cache_shape_dims:
        msg = f"key_cache tensor has unexpected shape ({key_cache.shape = }), expected {expected_kv_cache_shape_dims}-D tensor"
        raise ValueError(msg)

    if len(value_cache.shape) != expected_kv_cache_shape_dims:
        msg = f"value_cache tensor has unexpected shape ({value_cache.shape = }), expected {expected_kv_cache_shape_dims}-D tensor"
        raise ValueError(msg)

    if key_cache.shape != value_cache.shape:
        msg = f"Shape of key and value cache tensors do not match ({key_cache.shape = }, {value_cache.shape = })"
        raise ValueError(msg)

    if key_cache.size(-1) != head_size:
        msg = (
            f"Last dimension in key/value_cache shape ({key_cache.shape = }) does not match head_size ({head_size = })"
        )


def _check_block_table_size_compatibility(block_table: torch.Tensor, batch_size: int) -> None:
    """Check size compatibility of block_table tensor.

    Args:
        block_table: Block tables tensor.
        batch_size: Expected size of batch.

    Raises:
        ValueError if sizes are mismatched.
    """
    batch_size_block_table: int = block_table.shape[0]

    if batch_size_block_table != batch_size:
        msg = f"Batch size from block_table tensor ({batch_size_block_table}) does not match batch_size from output/query tensors ({batch_size})"
        raise ValueError(msg)


def _determine_max_num_kv_splits(max_seqlen_k: int) -> int:
    # Note: this is a pretty basic heuristic, we could try and do something more sophisticated in the future
    if max_seqlen_k > 8192:
        return 64

    if max_seqlen_k > 2048:
        return 32

    if max_seqlen_k > 1024:
        return 16

    if max_seqlen_k > 512:
        return 8

    if max_seqlen_k > 256:
        return 4

    if max_seqlen_k > 128:
        return 2

    return 1


def _check_size_compatibility(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
) -> PagedAttentionMetadata:
    """Check size compatibility of tensors for PagedAttention and return Metadata if successful.

    Args:
        out: Tensor to write the output of the attention calculation, shape: (batch_size, num_heads, head_size).
        query: Query tensor, shape: (batch_size, num_heads, head_size).
        key_cache: Tensor holding cache for K, shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        value_cache: Tensor holding cache for V, shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        block_table: Tensor storing the mapping from batch to cache blocks, shape: (batch_size, max_num_blocks_per_sequence).

    Raises:
        ValueError if sizes are mismatched.

    Returns:
        Metadata dataclass holding information about tensor sizes.
    """
    _check_output_query_size_compatibility(out, query)
    batch_size, num_query_heads, head_size = out.shape

    _check_kv_cache_size_compatibility(key_cache, value_cache, head_size)
    num_cache_blocks, cache_block_size, num_kv_heads, _ = key_cache.shape

    _check_block_table_size_compatibility(block_table, batch_size)
    _, max_num_blocks_per_sequence = block_table.shape

    return PagedAttentionMetadata(
        batch_size=batch_size,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        num_cache_blocks=num_cache_blocks,
        max_num_blocks_per_sequence=max_num_blocks_per_sequence,
        max_num_splits=_determine_max_num_kv_splits(max_num_blocks_per_sequence * cache_block_size),
    )


def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor | None = None,
    scale: float | None = None,
    softcap: float = 0.0,
    kv_cache_dtype: str = "auto",
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """PagedAttention interface to verify sizes and launch kernel.

    Args:
        query: Query tensor, shape: (batch_size, num_heads, head_size).
        key_cache: Tensor holding cache for K, shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        value_cache: Tensor holding cache for V, shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        block_table: Tensor storing the mapping from batch to cache blocks, shape: (batch_size, max_num_blocks_per_sequence).
        seq_lens: Tensor with the sequence length of each index in the batch, shape: (batch_size, ).
        output: Tensor to write the output of the attention calculation, shape: (batch_size, num_heads, head_size).
        scale: Scaling factor, 1/sqrt(head_size).
        softcap: Logit softcap to apply (0.0 means no softcap will be applied).
        kv_cache_dtype: Data type of the KV cache.
        k_scale: FP8 scaling factor for key cache.
        v_scale: FP8 scaling factor for value cache.
    """
    # Allocate output tensor if not provided
    if output is None:
        output = torch.zeros_like(query, device=query.device, dtype=query.dtype)

    # Check sizes of input tensors
    metadata = _check_size_compatibility(output, query, key_cache, value_cache, block_table)

    # Note: we could allocate this scratch outside of this function so that we could reuse it. vLLM allocates the scratch memory inline like this

    # Allocate additional memory for intermediate result (of shape (head_size,)) for each batch/query head/cache block
    output_scratchpad = torch.zeros(
        (metadata.batch_size, metadata.max_num_splits, metadata.num_query_heads, metadata.head_size),
        dtype=output.dtype,
        device=output.device,
    )

    # Allocate additional memory for intermediate log-sum-exp ("lse", scalar value per-cache block) for each batch/query head/cache block
    lse_scratchpad = torch.zeros(
        (metadata.batch_size, metadata.max_num_splits, metadata.num_query_heads),
        dtype=output.dtype,
        device=output.device,
    )

    paged_attention_launcher(
        output,
        query,
        key_cache,
        value_cache,
        output_scratchpad,
        lse_scratchpad,
        block_table,
        seq_lens,
        scale,
        softcap,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )

    return output
