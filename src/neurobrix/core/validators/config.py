"""
NeuroBrix Validators - Configuration.

Global configuration for validation behavior.
Thread-safe singleton pattern for consistent settings.

ZERO HARDCODING: All thresholds are configurable.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Set
import threading

from .base import SafetyLevel


@dataclass
class ValidatorConfig:
    """
    Configuration for all validators.

    Configurable thresholds for:
    - Memory limits
    - Numerical stability
    - Shape constraints
    - Performance budgets
    """

    # Safety level
    safety_level: SafetyLevel = SafetyLevel.STANDARD

    # Memory thresholds
    memory_safety_margin: float = 0.1  # 10% safety margin
    max_tensor_bytes: int = 16 * 1024**3  # 16 GB max single tensor
    warn_memory_usage: float = 0.8  # Warn at 80% usage

    # Numerical thresholds
    nan_threshold: int = 0  # Any NaN is an error
    inf_threshold: int = 0  # Any Inf is an error
    max_fp16_value: float = 65504.0  # FP16 max representable
    max_bf16_value: float = 3.38953e38  # BF16 max representable
    gradient_clip_threshold: float = 1.0  # Gradient clipping threshold

    # Shape constraints
    max_batch_size: int = 4096
    max_sequence_length: int = 1048576  # 1M tokens
    max_hidden_dim: int = 65536
    max_tensor_dims: int = 8  # Max number of dimensions

    # Sync thresholds
    max_stream_depth: int = 32  # Max concurrent streams
    sync_timeout_ms: int = 30000  # 30 second timeout

    # Fusion thresholds
    max_fusion_ops: int = 16  # Max ops in single fused kernel
    min_fusion_benefit: float = 0.1  # 10% speedup required

    # Supported operations (can be extended)
    supported_ops: Set[str] = field(default_factory=lambda: {
        # Core arithmetic
        "add", "sub", "mul", "div", "matmul", "linear",
        # Activations
        "relu", "gelu", "silu", "sigmoid", "tanh", "softmax",
        # Normalization
        "layernorm", "rmsnorm", "groupnorm", "batchnorm",
        # Reduction
        "sum", "mean", "max", "min",
        # Attention
        "scaled_dot_product_attention", "flash_attention",
        # Convolution
        "conv2d", "conv1d",
        # Pooling
        "maxpool2d", "avgpool2d",
        # Memory
        "reshape", "transpose", "permute", "view", "contiguous",
        "concat", "split", "slice", "gather", "scatter",
        # Embedding
        "embedding", "rotary_embedding",
        # Misc
        "dropout", "sqrt", "exp", "log", "pow",
        # ONNX CamelCase variants
        "scaleddotproductattention", "flashattention", "layernormalization",
    })

    # Performance budgets (optional)
    max_kernel_time_ms: Optional[float] = None
    max_memory_bandwidth_usage: Optional[float] = None

    def validate_memory_requirement(self, required_bytes: int, available_bytes: int) -> bool:
        """Check if memory requirement is within limits."""
        usable = available_bytes * (1.0 - self.memory_safety_margin)
        return required_bytes <= usable

    def is_op_supported(self, op: str) -> bool:
        """Check if operation is supported."""
        return op.lower() in self.supported_ops

    def get_dtype_max(self, dtype: str) -> float:
        """Get max representable value for dtype."""
        dtype_maxes = {
            "float16": self.max_fp16_value,
            "bfloat16": self.max_bf16_value,
            "float32": 3.4028235e38,
            "float64": 1.7976931348623157e308,
        }
        return dtype_maxes.get(dtype.lower(), float('inf'))


# ============================================================================
# Global Configuration Singleton
# ============================================================================

_config_lock = threading.Lock()
_global_config: Optional[ValidatorConfig] = None


def get_config() -> ValidatorConfig:
    """
    Get the global validator configuration.

    Thread-safe singleton pattern.
    """
    global _global_config

    if _global_config is None:
        with _config_lock:
            if _global_config is None:
                _global_config = ValidatorConfig()

    return _global_config


def set_config(config: ValidatorConfig) -> None:
    """
    Set the global validator configuration.

    Thread-safe.
    """
    global _global_config

    with _config_lock:
        _global_config = config


def reset_config() -> None:
    """Reset to default configuration."""
    global _global_config

    with _config_lock:
        _global_config = ValidatorConfig()


def update_config(**kwargs) -> ValidatorConfig:
    """
    Update specific configuration values.

    Returns the updated config.
    """
    config = get_config()

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config key: {key}")

    return config
