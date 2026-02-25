# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Flash Attention with varlen."""

from typing import Final

import torch
import triton

from conch.kernels.attention.varlen_attention import _FP8_DTYPES, varlen_attention_launcher


def _check_output_query_size_compatibility(output: torch.Tensor, query: torch.Tensor) -> None:
    """Check size compatibility of Output and Query tensors.

    Args:
        output: Tensor to write the output of the attention calculation, shape: (total_num_q, num_query_heads, head_size).
        query: Query tensor, shape: (total_num_q, num_query_heads, head_size).

    Raises:
        ValueError if sizes are mismatched.
    """
    # Output tensor should be a 3-D tensor of shape (total_num_q, num_query_heads, head_size)
    expected_output_shape_dims: Final = 3

    if len(output.shape) != expected_output_shape_dims:
        msg = f"Output tensor has unexpected shape ({output.shape = }), expected {expected_output_shape_dims}-D tensor"
        raise ValueError(msg)

    # Query tensor should have same shape as output
    if len(query.shape) != expected_output_shape_dims:
        msg = f"Query tensor has unexpected shape ({query.shape = }), expected {expected_output_shape_dims}-D tensor"
        raise ValueError(msg)

    if query.shape != output.shape:
        msg = f"Shape of query and output tensors does not match ({query.shape = }, {output.shape = })"
        raise ValueError(msg)


def _check_key_value_cache_size_compatibility(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    head_size: int,
    num_query_heads: int,
) -> None:
    """Check size compatibility of Key and Value tensors.

    Args:
        key_cache: Tensor with cached K values, shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        value_cache: Tensor with cached V values, shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        head_size: Size of attention head, deduced from query tensor size.
        num_query_heads: Number of query heads, deduced from query tensor size.

    Raises:
        ValueError if sizes are mismatched.
    """
    # Key/Value tensors should be 4-D tensors of shape (num_blocks, num_kv_heads, cache_block_size, head_size)
    expected_key_shape_dims: Final = 4

    if len(key_cache.shape) != expected_key_shape_dims:
        msg = (
            f"key_cache tensor has unexpected shape ({key_cache.shape = }), expected {expected_key_shape_dims}-D tensor"
        )
        raise ValueError(msg)

    if len(value_cache.shape) != expected_key_shape_dims:
        msg = f"value_cache tensor has unexpected shape ({value_cache.shape = }), expected {expected_key_shape_dims}-D tensor"
        raise ValueError(msg)

    if key_cache.shape != value_cache.shape:
        msg = (
            f"Shape of key_cache and value_cache tensors does not match ({key_cache.shape = }, {value_cache.shape = })"
        )
        raise ValueError(msg)

    num_blocks, cache_block_size, num_kv_heads, head_size_kv = key_cache.shape

    if head_size_kv != head_size:
        msg = f"Head size of key/value cache tensors does not match head size of query/output tensors ({head_size_kv = }, {head_size = })"
        raise ValueError(msg)

    if num_kv_heads > num_query_heads:
        msg = f"Number of key/value heads ({num_kv_heads}) is greater than number of query heads ({num_query_heads})"
        raise ValueError(msg)


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


def _check_seqlen_size_compatibility(seq_lens: torch.Tensor, batch_size: int) -> None:
    """Check size compatibility of seq_lens tensor.

    Args:
        seq_lens: Sequence lengths tensor.
        batch_size: Expected size of batch.

    Raises:
        ValueError if sizes are mismatched.
    """
    # Sequence lengths tensor should be 1-D tensors of shape (batch_size,)
    expected_seqlen_shape_dims: Final = 1

    if len(seq_lens.shape) != expected_seqlen_shape_dims:
        msg = f"Sequence lengths tensor has unexpected shape ({seq_lens.shape = }), expected {expected_seqlen_shape_dims}-D tensor"
        raise ValueError(msg)

    if seq_lens.shape[0] != batch_size:
        msg = f"Shape of sequence lengths tensor does not match batch size ({seq_lens.shape[0] = }, {batch_size = })"
        raise ValueError(msg)


def _determine_max_num_kv_splits(max_seqlen_q: int, max_seqlen_k: int, max_num_blocks_per_sequence: int) -> int:
    # If we have any prefills (max_seqlen_q > 1), disable FlashDecoding/KV-splits.
    # If we have all decodes, only enable FlashDecoding if we have a long sequence (>=4096 tokens) and
    # many cache blocks for each sequence (>=64 cache blocks for the longest sequence).
    # Note: this is a pretty basic heuristic, we could try and do something more sophisticated in the future. It'd
    # probably be more efficient to tune this per-platform, but we can't just autotune it because it determines how
    # much scratchpad memory we allocate. FlashAttention's heuristic uses the number of SMs, it might be worth
    # considering something like that, but preferably platform-agnostic.
    if max_seqlen_q == 1 and max_seqlen_k >= 4096 and max_num_blocks_per_sequence >= 64:
        # This number of KV splits affects the size of the scratchpad memory that we allocate,
        # so cap the number of splits at 64 so that we don't allocate too much memory
        return min(max_num_blocks_per_sequence // 16, 16)

    return 1


def _check_size_compatibility(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> None:
    """Check size compatibility of tensors for variable-length attention.

    Args:
        output: Tensor to write the output of the attention calculation, shape: (total_num_q, num_query_heads, head_size).
        query: Query tensor, shape: (total_num_q, num_query_heads, head_size).
        key_cache: Tensor with cached K values, shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        value_cache: Tensor with cached V values, shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        cu_seqlens_q: Cumulative sequence length for query/output tensors, shape: (batch_size + 1).
        seq_lens: Sequence lengths tensor, shape: (batch_size,).
        block_table: Block tables tensor, shape: (batch_size, max_num_blocks_per_sequence).
        max_seqlen_q: Maximum sequence length for query/output tensors.
        max_seqlen_k: Maximum sequence length for key/value tensors.

    Raises:
        ValueError if sizes are mismatched.
    """
    _check_output_query_size_compatibility(out, query)

    _, num_query_heads, head_size = out.shape
    batch_size = cu_seqlens_q.shape[0] - 1

    _check_key_value_cache_size_compatibility(key_cache, value_cache, head_size, num_query_heads)
    _check_block_table_size_compatibility(block_table, batch_size)
    _check_seqlen_size_compatibility(seq_lens, batch_size)


def varlen_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    seq_lens: torch.Tensor,
    max_seqlen_k: int,
    block_table: torch.Tensor,
    output: torch.Tensor | None = None,
    causal: bool = False,
    scale: float | None = None,
    softcap: float = 0.0,
    kv_cache_dtype: str = "auto",
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    strict: bool = False,
) -> torch.Tensor:
    """Varlen attention interface to verify sizes and launch kernel.

    Args:
        query: Query tensor, shape: (total_num_q, num_query_heads, head_size).
        key_cache: Tensor with cached K values, shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        value_cache: Tensor with cached V values, shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        cu_seqlens_q: Cumulative sequence length for query/output tensors, shape: (batch_size + 1).
        max_seqlen_q: Maximum sequence length for query/output tensors.
        seq_lens: Sequence lengths tensor, shape: (batch_size,).
        max_seqlen_k: Maximum sequence length for key/value tensors.
        block_table: Block tables tensor, shape: (batch_size, max_num_blocks_per_sequence).
        output: (Optional), Tensor to write the output of the attention calculation, shape: (total_num_q, num_query_heads, head_size).
        causal: (Optional), Whether to apply causal masking.
        scale: (Optional), Scaling factor, 1/sqrt(head_size).
        softcap: (Optional), Logit softcap to apply (0.0 means no softcap will be applied).
        kv_cache_dtype: (Optional), String datatype of KV-cache.
        q_scale: (Optional), Scaling factor for Q values.
        k_scale: (Optional), Scaling factor for K values.
        v_scale: (Optional), Scaling factor for V values.
        strict: (Optional), Enable strict checking of tensor sizes.
    """
    # Allocate output tensor if not provided
    if output is None:
        output_dtype = torch.bfloat16 if query.dtype in _FP8_DTYPES else query.dtype
        output = torch.zeros_like(query, device=query.device, dtype=output_dtype)

    # Check sizes of input tensors
    if strict:
        _check_size_compatibility(
            output,
            query,
            key_cache,
            value_cache,
            cu_seqlens_q,
            seq_lens,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
        )

    num_kv_splits = _determine_max_num_kv_splits(
        max_seqlen_q, max_seqlen_k, triton.cdiv(max_seqlen_k, key_cache.size(1))
    )

    output_scratchpad = None
    lse_scratchpad = None

    if num_kv_splits > 1:
        total_num_q = output.size(0)
        num_query_heads = output.size(1)
        head_size = output.size(2)

        # Allocate additional memory for intermediate result (of shape (head_size,)) for each batch/kv split/query head
        output_scratchpad = torch.empty(
            (num_kv_splits, total_num_q, num_query_heads, head_size),
            dtype=output.dtype,
            device=output.device,
        )

        # Allocate additional memory for intermediate log-sum-exp ("lse", scalar value per-cache block) for each batch/kv split/query head
        lse_scratchpad = torch.empty(
            (num_kv_splits, total_num_q, num_query_heads),
            dtype=output.dtype,
            device=output.device,
        )

    varlen_attention_launcher(
        output=output,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        seq_lens=seq_lens,
        max_seqlen_k=max_seqlen_k,
        block_table=block_table,
        output_scratchpad=output_scratchpad,
        lse_scratchpad=lse_scratchpad,
        causal=causal,
        scale=scale,
        softcap=softcap,
        kv_cache_dtype=kv_cache_dtype,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        strict=strict,
    )

    return output
