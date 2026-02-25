"""
NeuroBrix Validators - Runtime Guards.

Fast runtime checks for hot path validation.

ZERO FALLBACK: Runtime violations raise explicit exceptions.
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
import functools

from .base import (
    SafetyLevel, ValidationError, ShapeViolation,
    MemoryPlanError, NaNInfDetected
)
from .config import get_config


class RuntimeGuards:
    """
    Fast runtime validation guards.

    Designed for hot path with minimal overhead at MINIMAL level.
    More thorough checks at STANDARD and PARANOID levels.
    """

    def __init__(self, safety_level: SafetyLevel = SafetyLevel.STANDARD):
        self.safety_level = safety_level
        self.config = get_config()
        self._check_count = 0
        self._violation_count = 0

    # ========================================================================
    # Shape Guards
    # ========================================================================

    def check_shape(
        self,
        tensor: Any,
        expected_shape: Tuple[int, ...],
        name: str = "tensor",
    ) -> None:
        """
        Check tensor shape matches expected.

        Fast: O(n) where n = number of dimensions.
        """
        self._check_count += 1

        actual_shape = self._get_shape(tensor)
        if actual_shape != expected_shape:
            self._violation_count += 1
            raise ShapeViolation(
                f"Shape mismatch for {name}",
                expected_shape=expected_shape,
                actual_shape=actual_shape,
            )

    def check_shape_compatible(
        self,
        tensor_a: Any,
        tensor_b: Any,
        name_a: str = "tensor_a",
        name_b: str = "tensor_b",
    ) -> None:
        """
        Check two tensors have compatible shapes for operations.

        Checks broadcastability.
        """
        self._check_count += 1

        shape_a = self._get_shape(tensor_a)
        shape_b = self._get_shape(tensor_b)

        if not self._are_broadcastable(shape_a, shape_b):
            self._violation_count += 1
            raise ShapeViolation(
                f"Shapes not compatible: {name_a} and {name_b}",
                expected_shape=shape_a,
                actual_shape=shape_b,
                suggestion="Shapes must be broadcastable",
            )

    def check_batch_size(
        self,
        tensor: Any,
        max_batch: Optional[int] = None,
        name: str = "tensor",
    ) -> None:
        """Check batch size is within limits."""
        self._check_count += 1

        shape = self._get_shape(tensor)
        if not shape:
            return

        batch_size = shape[0]
        max_allowed = max_batch or self.config.max_batch_size

        if batch_size > max_allowed:
            self._violation_count += 1
            raise ShapeViolation(
                f"Batch size {batch_size} exceeds max {max_allowed} for {name}",
                actual_shape=shape,
                suggestion=f"Reduce batch size to at most {max_allowed}",
            )

    def _get_shape(self, tensor: Any) -> Tuple[int, ...]:
        """Get tensor shape."""
        if hasattr(tensor, 'shape'):
            return tuple(tensor.shape)
        elif hasattr(tensor, 'size'):
            return tuple(tensor.size())
        return ()

    def _are_broadcastable(
        self,
        shape1: Tuple[int, ...],
        shape2: Tuple[int, ...],
    ) -> bool:
        """Check if shapes are broadcastable."""
        len1, len2 = len(shape1), len(shape2)
        max_len = max(len1, len2)

        for i in range(1, max_len + 1):
            d1 = shape1[-i] if i <= len1 else 1
            d2 = shape2[-i] if i <= len2 else 1

            if d1 != d2 and d1 != 1 and d2 != 1:
                return False

        return True

    # ========================================================================
    # Numerical Guards
    # ========================================================================

    def check_no_nan(
        self,
        tensor: Any,
        name: str = "tensor",
    ) -> None:
        """
        Check tensor contains no NaN values.

        Only runs at STANDARD+ levels by default.
        """
        if self.safety_level == SafetyLevel.MINIMAL:
            return

        self._check_count += 1

        if hasattr(tensor, 'isnan'):
            has_nan = tensor.isnan().any().item()
            if has_nan:
                self._violation_count += 1
                nan_count = int(tensor.isnan().sum().item())
                raise NaNInfDetected(
                    f"NaN detected in {name}",
                    nan_count=nan_count,
                    suggestion="Check for division by zero or log of negative values",
                )

    def check_no_inf(
        self,
        tensor: Any,
        name: str = "tensor",
    ) -> None:
        """
        Check tensor contains no Inf values.

        Only runs at STANDARD+ levels by default.
        """
        if self.safety_level == SafetyLevel.MINIMAL:
            return

        self._check_count += 1

        if hasattr(tensor, 'isinf'):
            has_inf = tensor.isinf().any().item()
            if has_inf:
                self._violation_count += 1
                inf_count = int(tensor.isinf().sum().item())
                raise NaNInfDetected(
                    f"Inf detected in {name}",
                    inf_count=inf_count,
                    suggestion="Check for overflow or very large intermediate values",
                )

    def check_finite(
        self,
        tensor: Any,
        name: str = "tensor",
    ) -> None:
        """Check tensor contains only finite values (no NaN or Inf)."""
        self.check_no_nan(tensor, name)
        self.check_no_inf(tensor, name)

    def check_value_range(
        self,
        tensor: Any,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        name: str = "tensor",
    ) -> None:
        """Check tensor values are within range."""
        if self.safety_level == SafetyLevel.MINIMAL:
            return

        self._check_count += 1

        if hasattr(tensor, 'min') and hasattr(tensor, 'max'):
            actual_min = float(tensor.min().item())
            actual_max = float(tensor.max().item())

            if min_val is not None and actual_min < min_val:
                self._violation_count += 1
                raise ValidationError(
                    f"Value below minimum in {name}: {actual_min} < {min_val}",
                    context={'actual_min': actual_min, 'required_min': min_val},
                )

            if max_val is not None and actual_max > max_val:
                self._violation_count += 1
                raise ValidationError(
                    f"Value above maximum in {name}: {actual_max} > {max_val}",
                    context={'actual_max': actual_max, 'required_max': max_val},
                )

    # ========================================================================
    # Memory Guards
    # ========================================================================

    def check_memory_available(
        self,
        required_bytes: int,
        available_bytes: int,
        name: str = "allocation",
    ) -> None:
        """Check sufficient memory is available."""
        self._check_count += 1

        config = self.config
        usable = int(available_bytes * (1.0 - config.memory_safety_margin))

        if required_bytes > usable:
            self._violation_count += 1
            raise MemoryPlanError(
                f"Insufficient memory for {name}",
                required_bytes=required_bytes,
                available_bytes=available_bytes,
                suggestion="Consider gradient checkpointing or smaller batch size",
            )

    def check_tensor_size(
        self,
        tensor: Any,
        max_bytes: Optional[int] = None,
        name: str = "tensor",
    ) -> None:
        """Check tensor size is within limits."""
        self._check_count += 1

        # Estimate tensor size
        size_bytes = self._estimate_tensor_bytes(tensor)
        max_allowed = max_bytes or self.config.max_tensor_bytes

        if size_bytes > max_allowed:
            self._violation_count += 1
            raise MemoryPlanError(
                f"Tensor {name} too large: {size_bytes / 1024**3:.2f} GB",
                required_bytes=size_bytes,
                suggestion=f"Max tensor size is {max_allowed / 1024**3:.0f} GB",
            )

    def _estimate_tensor_bytes(self, tensor: Any) -> int:
        """Estimate tensor size in bytes."""
        if hasattr(tensor, 'nbytes'):
            return tensor.nbytes
        elif hasattr(tensor, 'numel') and hasattr(tensor, 'element_size'):
            return tensor.numel() * tensor.element_size()
        elif hasattr(tensor, 'shape'):
            # Assume float32
            import functools
            import operator
            numel = functools.reduce(operator.mul, tensor.shape, 1)
            return numel * 4
        return 0

    # ========================================================================
    # Decorators
    # ========================================================================

    def guard(
        self,
        check_shapes: bool = True,
        check_numerical: bool = True,
        check_memory: bool = False,
    ) -> Callable:
        """
        Decorator to add runtime guards to a function.

        Usage:
            @guards.guard(check_shapes=True, check_numerical=True)
            def my_kernel(x, y):
                ...
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Pre-checks
                if check_shapes and args:
                    for i, arg in enumerate(args):
                        if hasattr(arg, 'shape'):
                            self.check_batch_size(arg, name=f"arg_{i}")

                # Execute function
                result = func(*args, **kwargs)

                # Post-checks
                if check_numerical and hasattr(result, 'isnan'):
                    self.check_finite(result, name=f"{func.__name__}_output")

                return result
            return wrapper
        return decorator

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get runtime guard statistics."""
        return {
            'check_count': self._check_count,
            'violation_count': self._violation_count,
            'safety_level': self.safety_level.name,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._check_count = 0
        self._violation_count = 0
