"""Triton execution strategies — NBXTensor-native (zero torch).

Mirror of the PyTorch strategy registry for the triton execution path.
`get_strategy()` (parent package) routes here when `context.mode` is
"triton" / "triton_sequential". Only MIGRATED strategies appear here; for
any name absent from `TRITON_REGISTRY`, get_strategy falls back to the
PyTorch class (which already runs on the triton path via the polymorphic
transfer helper) — so migration is incremental and regression-free.
"""

from .base import TritonStrategy
from .single_gpu import SingleGPUStrategy

# Triton-native strategies, keyed by the same Prism strategy names as the
# PyTorch registry. Add entries here as each strategy is ported.
TRITON_REGISTRY = {
    "single_gpu": SingleGPUStrategy,
    "single_gpu_lifecycle": SingleGPUStrategy,
}

__all__ = ["TritonStrategy", "SingleGPUStrategy", "TRITON_REGISTRY"]
