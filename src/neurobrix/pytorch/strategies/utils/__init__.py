"""
Strategy Utilities Package

Shared utilities for execution strategies.
"""

from .block_detector import (
    extract_block_index,
    extract_block_indices_from_paths,
    parse_block_structure,
    BLOCK_PATTERNS,
)

__all__ = [
    "extract_block_index",
    "extract_block_indices_from_paths",
    "parse_block_structure",
    "BLOCK_PATTERNS",
]
