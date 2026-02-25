"""
Block Detection Utilities

Shared utilities for detecting block structure from weight names.
Used by Pipeline, TensorParallel, and Zero3 strategies.

ZERO HARDCODE: Pattern-based detection from weight keys.
"""

import re
from typing import Dict, List, Optional, Set


# Common block naming patterns across model architectures
BLOCK_PATTERNS = [
    r'blocks?[._](\d+)',              # blocks.0, block_0, block.1
    r'layers?[._](\d+)',              # layer.0, layers_0, layer.1
    r'transformer[._]h[._](\d+)',     # transformer.h.0 (GPT style)
    r'encoder[._]layer[._](\d+)',     # encoder.layer.0 (BERT style)
    r'decoder[._]layer[._](\d+)',     # decoder.layer.0
    r'model[._]layers[._](\d+)',      # model.layers.0 (LLaMA style)
    r'resblocks[._](\d+)',            # resblocks.0 (CLIP style)
    r'down_blocks[._](\d+)',          # down_blocks.0 (UNet style)
    r'up_blocks[._](\d+)',            # up_blocks.0 (UNet style)
    r'mid_block[._](\d+)',            # mid_block.0 (UNet style)
    r'joint_blocks[._](\d+)',         # joint_blocks.0 (DiT style)
    r'single_transformer_blocks[._](\d+)',  # Flux style
    r'transformer_blocks[._](\d+)',   # transformer_blocks.0
]


def extract_block_index(weight_name: str) -> Optional[int]:
    """
    Extract block index from a weight name.

    Args:
        weight_name: Full weight key (e.g., "transformer.blocks.5.attn.qkv.weight")

    Returns:
        Block index (e.g., 5) or None if no block pattern found

    Example:
        >>> extract_block_index("transformer.blocks.5.attn.qkv.weight")
        5
        >>> extract_block_index("embed_tokens.weight")
        None
    """
    for pattern in BLOCK_PATTERNS:
        match = re.search(pattern, weight_name, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def extract_block_indices_from_paths(paths: List[str]) -> List[int]:
    """
    Extract all unique block indices from a list of weight paths.

    Args:
        paths: List of weight paths/keys

    Returns:
        Sorted list of unique block indices found

    Example:
        >>> paths = ["blocks.0.attn.weight", "blocks.1.attn.weight", "embed.weight"]
        >>> extract_block_indices_from_paths(paths)
        [0, 1]
    """
    indices: Set[int] = set()

    for path in paths:
        idx = extract_block_index(path)
        if idx is not None:
            indices.add(idx)

    return sorted(indices)


def parse_block_structure(
    weight_keys: List[str],
    include_non_block: bool = True,
) -> Dict[int, List[str]]:
    """
    Parse weight keys into block-organized structure.

    Groups weights by their block index. Non-block weights (embeddings,
    final layers) are optionally assigned to block -1.

    Args:
        weight_keys: List of all weight keys
        include_non_block: If True, non-block weights go to key -1

    Returns:
        Dict mapping block_index -> list of weight keys
        Block -1 contains non-block weights if include_non_block=True

    Example:
        >>> keys = ["blocks.0.attn.weight", "blocks.1.attn.weight", "embed.weight"]
        >>> structure = parse_block_structure(keys)
        >>> structure[0]
        ['blocks.0.attn.weight']
        >>> structure[-1]  # Non-block weights
        ['embed.weight']
    """
    block_weights: Dict[int, List[str]] = {}
    non_block_weights: List[str] = []

    for key in weight_keys:
        block_idx = extract_block_index(key)

        if block_idx is not None:
            if block_idx not in block_weights:
                block_weights[block_idx] = []
            block_weights[block_idx].append(key)
        else:
            non_block_weights.append(key)

    # Add non-block weights to special key -1
    if include_non_block and non_block_weights:
        block_weights[-1] = non_block_weights

    return block_weights


def classify_non_block_weight(weight_name: str) -> str:
    """
    Classify a non-block weight as 'early' or 'late' in the model.

    Early weights (embeddings, input layers) should be processed first.
    Late weights (output layers, final norms) should be processed last.

    Args:
        weight_name: Weight key without block index

    Returns:
        'early' or 'late'
    """
    name_lower = weight_name.lower()

    # Early layers (input side)
    early_hints = [
        'embed', 'patch', 'pos', 'input', 'norm_pre', 'time_',
        'conv_in', 'proj_in', 'x_embedder', 't_embedder',
        'text_proj', 'context_embedder', 'adaln_single',
    ]

    for hint in early_hints:
        if hint in name_lower:
            return 'early'

    # Everything else is late (output side)
    return 'late'


def get_block_count(weight_keys: List[str]) -> int:
    """
    Get the number of unique blocks in weight keys.

    Args:
        weight_keys: List of weight keys

    Returns:
        Number of unique blocks (excluding non-block weights)
    """
    indices = extract_block_indices_from_paths(weight_keys)
    return len(indices)
