# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Rotary Embedding."""

import torch

from conch.kernels.embedding.rotary_embedding import rotary_embedding_launcher


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    *,
    is_neox: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding to query and key.

    NOTE: Only supporting NeoX model.

    Args:
        positions: token positions [num_tokens].
        query: queries [num_tokens, num_heads * head_size].
        key: keys [num_tokens, num_heads * head_size].
        head_size: head size.
        cos_sin_cache: precacluated cos-sin, typically [rot_dim], split into [cos | sin] for NeoX; use tensor provided by vllm::RotaryEmbedding.
        is_neox: whether using NeoX, must be True; exists mostly for visibility.
    """
    rotary_embedding_launcher(positions, query, key, head_size, cos_sin_cache, is_neox=is_neox)
    return query, key
