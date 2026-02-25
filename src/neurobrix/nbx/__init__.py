"""
NBX Format Module - Universal Neural Network Container Format

This module implements the NBX format specification v1.0:
- Create and read .nbx container files
- Universal operation mapping
- NeuroTax: Universal tensor naming taxonomy
- Fast cache-based loading with direct GPU transfer

PERFORMANCE OPTIMIZATION (Fast Loading):
  OLD: ZIP -> CPU RAM -> GPU = 3 copies, ~108s for 21GB
  NEW: ZIP -> Extract once -> Direct GPU load = 1 copy, ~30s
"""

from .spec import (
    NBX_VERSION,
    NBX_MAGIC,
    ONNX_TO_NBX_OPS,
    NBX_DTYPES,
    get_nbx_op,
    get_nbx_dtype,
)

from .container import NBXContainer, ComponentData

# NeuroTax - Universal tensor naming taxonomy
from .neurotax import (
    SynonymRegistry,
    ParsedTensor,
    NeuroTaxParser,
    normalize_tensor_name,
    normalize_tensor_name_strict,
)

# Fast Loading - Cache + Direct GPU
from .cache import NBXCache, get_cache, ensure_extracted
from .loader import FastNBXLoader, fast_load_nbx, fast_load_weights

__all__ = [
    # Version and constants
    "NBX_VERSION",
    "NBX_MAGIC",
    "ONNX_TO_NBX_OPS",
    "NBX_DTYPES",
    # Helper functions
    "get_nbx_op",
    "get_nbx_dtype",
    # Classes
    "NBXContainer",
    "ComponentData",
    # NeuroTax
    "SynonymRegistry",
    "ParsedTensor",
    "NeuroTaxParser",
    "normalize_tensor_name",
    "normalize_tensor_name_strict",
    # Fast Loading
    "NBXCache",
    "get_cache",
    "ensure_extracted",
    "FastNBXLoader",
    "fast_load_nbx",
    "fast_load_weights",
]
