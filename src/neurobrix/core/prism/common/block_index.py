# core/prism/common/block_index.py
"""
Block Index Extractor for Prism

Extracts block indices from shard filenames for pipeline parallel allocation.

ZERO HARDCODE: Uses regex patterns, not hardcoded indices.
"""

import re
from typing import Optional, List, Tuple


# Supported filename patterns for block extraction
BLOCK_PATTERNS = [
    r'block[_.](\d+)',      # block_0003, block.3
    r'blocks[_.](\d+)',     # blocks_0003
    r'layer[_.](\d+)',      # layer_3
    r'layers[_.](\d+)',     # layers_3
    r'transformer[_.](\d+)', # transformer_0, transformer.5
]


class BlockIndexExtractor:
    """
    Extracts block indices from shard filenames.
    
    Pipeline parallelism requires blocks to be placed sequentially on GPUs:
    - GPU0: blocks 0, 1, 2 (sequential)
    - GPU1: blocks 3, 4, 5 (sequential)
    NOT: GPU0: blocks 0, 2, 4 (interleaved - would break pipeline)
    """
    
    def __init__(self, additional_patterns: Optional[List[str]] = None):
        """
        Initialize extractor with patterns.
        
        Args:
            additional_patterns: Additional regex patterns for block extraction
        """
        self.patterns = BLOCK_PATTERNS.copy()
        if additional_patterns:
            self.patterns.extend(additional_patterns)
            
    def extract(self, shard_path: str) -> Optional[int]:
        """
        Extract block index from shard filename.
        
        Args:
            shard_path: Full path to shard file
            
        Returns:
            Block index or None if no pattern matches
        """
        filename = shard_path.split('/')[-1]
        
        for pattern in self.patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return int(match.group(1))
                
        return None
        
    def sort_by_block(
        self,
        shard_paths: List[str],
    ) -> List[Tuple[str, int]]:
        """
        Sort shard paths by block index.
        
        Non-block shards (e.g., embeddings) get index -1 to be placed first.
        
        Args:
            shard_paths: List of shard file paths
            
        Returns:
            List of (path, block_index) tuples sorted by block index
        """
        shards_with_blocks = []
        
        for path in shard_paths:
            block_idx = self.extract(path)
            if block_idx is None:
                block_idx = -1  # Non-block shards go first
            shards_with_blocks.append((path, block_idx))
            
        # Sort by block index, then by path for determinism
        shards_with_blocks.sort(key=lambda x: (x[1], x[0]))
        
        return shards_with_blocks


def extract_block_index(shard_path: str) -> Optional[int]:
    """
    Convenience function to extract block index.
    
    Args:
        shard_path: Path to shard file
        
    Returns:
        Block index or None
    """
    extractor = BlockIndexExtractor()
    return extractor.extract(shard_path)
