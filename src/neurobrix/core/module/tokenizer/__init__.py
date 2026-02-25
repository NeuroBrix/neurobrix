# core/module/tokenizer/__init__.py
"""
Tokenizer Module.

Text tokenization utilities for LLM and multimodal models.

Classes:
- SPTokenizer: SentencePiece tokenizer wrapper (T5/FLAN)
- BPETokenizer: Byte-Pair Encoding tokenizer (CLIP)
- HFTokenizer: HuggingFace fast tokenizer wrapper
- TokenizerWrapper: Wrapper for HF tokenizers
- TokenizerFactory: Factory for creating tokenizers from NBX cache

Functions:
- load_tokenizer_from_path: Load tokenizer with automatic type detection
- load_tokenizer_from_nbx: Load tokenizer from NBX container
"""

from neurobrix.core.module.tokenizer.sp_tokenizer import (
    SPTokenizer,
    BPETokenizer,
    HFTokenizer,
    load_tokenizer_from_path,
    load_tokenizer_from_nbx,
)
from neurobrix.core.module.tokenizer.factory import (
    TokenizerWrapper,
    TokenizerFactory,
)

__all__ = [
    # Tokenizer implementations
    "SPTokenizer",
    "BPETokenizer",
    "HFTokenizer",
    # Factory and wrapper
    "TokenizerWrapper",
    "TokenizerFactory",
    # Loading functions
    "load_tokenizer_from_path",
    "load_tokenizer_from_nbx",
]
