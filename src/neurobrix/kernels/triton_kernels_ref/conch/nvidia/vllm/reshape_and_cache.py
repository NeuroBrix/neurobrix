# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""vLLM reshape_and_cache."""

from typing import Final

import torch

from conch.kernels.vllm.reshape_and_cache import reshape_and_cache_launcher


def _validate_sizes(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Validate sizes of input tensors.

    Args:
        key: New key vectors, shape: (num_tokens, num_kv_heads, head_size).
        value: New value vectors, shape: (num_tokens, num_kv_heads, head_size).
        key_cache: Key cache, shape: (num_pages, cache_block_size, num_kv_heads, head_size).
        value_cache: Value cache, shape: (num_pages, cache_block_size, num_kv_heads, head_size).
        slot_mapping: Tensor listing what slots in the cache each key/value vector should be placed in, shape: (num_tokens,).

    Raises:
        ValueError if sizes are mismatched between various input tensors.
    """
    if key.shape != value.shape:
        msg = f"key.shape ({key.shape}) does not match value.shape ({value.shape})"
        raise ValueError(msg)

    expected_kv_dims: Final = 3
    if (key_dims := len(key.shape)) != expected_kv_dims:
        msg = f"Number of dimensions in key ({key_dims}) did not match expected ({expected_kv_dims})"
        raise ValueError(msg)

    _, num_kv_heads_kv, head_size_kv = key.shape

    if key_cache.shape != value_cache.shape:
        msg = f"key_cache.shape ({key_cache.shape}) does not match value_cache.shape ({value_cache.shape})"
        raise ValueError(msg)

    expected_kv_cache_dims: Final = 4
    if (key_cache_dims := len(key_cache.shape)) != expected_kv_cache_dims:
        msg = f"Number of dimensions in key cache ({key_cache_dims}) did not match expected ({expected_kv_cache_dims})"
        raise ValueError(msg)

    _, _, num_kv_heads_kvc, head_size_kvc = key_cache.shape

    if num_kv_heads_kv != num_kv_heads_kvc:
        msg = f"Number of kv heads in key/value tensors ({num_kv_heads_kv}) does not match number of kv heads in key/value cache tensors ({num_kv_heads_kvc})"
        raise ValueError(msg)

    if head_size_kv != head_size_kvc:
        msg = f"Head size in key/value tensors ({head_size_kv}) does not match head size in key/value cache tensors ({head_size_kvc})"
        raise ValueError(msg)

    expected_slot_mapping_dims: Final = 1
    if (slot_mapping_dims := len(slot_mapping.shape)) != expected_slot_mapping_dims:
        msg = f"Number of dimensions in slot mapping ({slot_mapping_dims}) did not match expected ({expected_slot_mapping_dims})"
        raise ValueError(msg)


def _validate_kv_cache_dtype(kv_cache_dtype: str) -> None:
    """Validate that KV Cache Dtype is valid and return whether to enable FP8 scaling.

    Args:
        kv_cache_dtype: String representing desired datatype of KV-cache.

    Raises:
        ValueError if kv_cache_dtype is invalid.
    """
    fp8_dtypes: Final = {"fp8", "fp8_e4m3"}
    allowed_dtypes: Final = {"auto"}.union(fp8_dtypes)

    if kv_cache_dtype not in allowed_dtypes:
        msg = f"Unsupported kv_cache_dtype: '{kv_cache_dtype}'"
        raise ValueError(msg)


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str = "auto",
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    strict: bool = False,
) -> None:
    """Reshape key/value vectors and add them to the cache.

    Args:
        key: New key vectors, shape: (num_tokens, num_kv_heads, head_size).
        value: New value vectors, shape: (num_tokens, num_kv_heads, head_size).
        key_cache: Key cache, shape: (num_pages, cache_block_size, num_kv_heads, head_size).
        value_cache: Value cache, shape: (num_pages, cache_block_size, num_kv_heads, head_size).
        slot_mapping: Tensor listing what slots in the cache each key/value vector should be placed in, shape: (num_tokens,).
        kv_cache_dtype: String datatype of kv cache elements.
        k_scale: Fp8 scaling factor for k.
        v_scale: Fp8 scaling factor for v.
    """
    if strict:
        # Verify input sizes/tensor shapes
        _validate_sizes(key, value, key_cache, value_cache, slot_mapping)

        # Validate kv cache dtype is valid
        _validate_kv_cache_dtype(kv_cache_dtype)

    # Call kernel launch wrapper
    reshape_and_cache_launcher(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
        strict=strict,
    )
