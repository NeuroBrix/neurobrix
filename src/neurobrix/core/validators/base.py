"""
NeuroBrix Validators - Base Classes.

Defines the foundation for all validation in NeuroBrix:
- SafetyLevel enum for validation intensity
- Base exceptions with rich context
- ValidationResult for structured reporting
- BaseValidator abstract class

ZERO FALLBACK: All validation failures raise explicit exceptions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # Future: from ..graph import GraphNode


class SafetyLevel(IntEnum):
    """
    Validation intensity levels.

    MINIMAL: Fast checks only (production hot path)
    STANDARD: Balanced safety/performance (default)
    PARANOID: All checks enabled (debugging/development)

    Uses IntEnum to support comparison operators (< > <= >=).
    """
    MINIMAL = 1
    STANDARD = 2
    PARANOID = 3


# ============================================================================
# Exceptions
# ============================================================================

class ValidationError(Exception):
    """
    Base exception for all validation failures.

    All validation exceptions carry rich context for debugging.
    """

    def __init__(
        self,
        message: str,
        *,
        node: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
    ):
        self.node = node
        self.context = context or {}
        self.suggestion = suggestion

        # Build rich error message
        parts = [message]
        if node is not None:
            node_name = getattr(node, 'name', str(node))
            node_op = getattr(node, 'op', 'unknown')
            parts.append(f"Node: {node_name} (op={node_op})")
        if context:
            parts.append(f"Context: {context}")
        if suggestion:
            parts.append(f"Suggestion: {suggestion}")

        super().__init__("\n".join(parts))


class ShapeViolation(ValidationError):
    """Raised when tensor shapes are incompatible."""

    def __init__(
        self,
        message: str,
        *,
        expected_shape: Optional[tuple] = None,
        actual_shape: Optional[tuple] = None,
        **kwargs
    ):
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape

        context = kwargs.pop('context', {})
        if expected_shape is not None:
            context['expected_shape'] = expected_shape
        if actual_shape is not None:
            context['actual_shape'] = actual_shape

        super().__init__(message, context=context, **kwargs)


class MemoryPlanError(ValidationError):
    """Raised when memory plan is invalid or exceeds limits."""

    def __init__(
        self,
        message: str,
        *,
        required_bytes: Optional[int] = None,
        available_bytes: Optional[int] = None,
        **kwargs
    ):
        self.required_bytes = required_bytes
        self.available_bytes = available_bytes

        context = kwargs.pop('context', {})
        if required_bytes is not None:
            context['required_bytes'] = required_bytes
            context['required_gb'] = required_bytes / (1024**3)
        if available_bytes is not None:
            context['available_bytes'] = available_bytes
            context['available_gb'] = available_bytes / (1024**3)

        super().__init__(message, context=context, **kwargs)


class NumericalDivergence(ValidationError):
    """Raised when numerical instability is detected."""

    def __init__(
        self,
        message: str,
        *,
        nan_count: Optional[int] = None,
        inf_count: Optional[int] = None,
        max_value: Optional[float] = None,
        **kwargs
    ):
        self.nan_count = nan_count
        self.inf_count = inf_count
        self.max_value = max_value

        context = kwargs.pop('context', {})
        if nan_count is not None:
            context['nan_count'] = nan_count
        if inf_count is not None:
            context['inf_count'] = inf_count
        if max_value is not None:
            context['max_value'] = max_value

        super().__init__(message, context=context, **kwargs)


class UnsupportedOpsError(ValidationError):
    """Raised when graph contains unsupported operations."""

    def __init__(
        self,
        message: str,
        *,
        unsupported_ops: Optional[List[str]] = None,
        **kwargs
    ):
        self.unsupported_ops = unsupported_ops or []

        context = kwargs.pop('context', {})
        if unsupported_ops:
            context['unsupported_ops'] = unsupported_ops

        super().__init__(message, context=context, **kwargs)


class SyncError(ValidationError):
    """Raised when synchronization issues are detected."""

    def __init__(
        self,
        message: str,
        *,
        device_a: Optional[str] = None,
        device_b: Optional[str] = None,
        **kwargs
    ):
        self.device_a = device_a
        self.device_b = device_b

        context = kwargs.pop('context', {})
        if device_a is not None:
            context['device_a'] = device_a
        if device_b is not None:
            context['device_b'] = device_b

        super().__init__(message, context=context, **kwargs)


class FusionError(ValidationError):
    """Raised when kernel fusion is invalid."""

    def __init__(
        self,
        message: str,
        *,
        nodes_to_fuse: Optional[List[str]] = None,
        reason: Optional[str] = None,
        **kwargs
    ):
        self.nodes_to_fuse = nodes_to_fuse or []
        self.reason = reason

        context = kwargs.pop('context', {})
        if nodes_to_fuse:
            context['nodes_to_fuse'] = nodes_to_fuse
        if reason:
            context['fusion_failure_reason'] = reason

        super().__init__(message, context=context, **kwargs)


class NaNInfDetected(NumericalDivergence):
    """Specific exception for NaN/Inf detection."""
    pass


class DeadlockRisk(SyncError):
    """Specific exception for potential deadlock situations."""
    pass


# ============================================================================
# Validation Result
# ============================================================================

@dataclass
class ValidationResult:
    """
    Structured result from validation.

    Contains pass/fail status plus detailed diagnostics.
    """
    passed: bool
    validator_name: str
    message: str = ""
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed

    def raise_if_failed(self) -> None:
        """Raise ValidationError if validation failed."""
        if not self.passed:
            raise ValidationError(
                f"Validation failed: {self.validator_name}\n{self.message}"
            )


# ============================================================================
# Base Validator
# ============================================================================

class BaseValidator(ABC):
    """
    Abstract base class for all validators.

    Validators check specific aspects of the execution graph:
    - Shapes compatibility
    - Memory feasibility
    - Numerical stability
    - Operation support
    - Synchronization correctness
    - Fusion validity
    """

    def __init__(self, safety_level: SafetyLevel = SafetyLevel.STANDARD):
        self.safety_level = safety_level

    @property
    @abstractmethod
    def name(self) -> str:
        """Validator name for reporting."""
        pass

    @abstractmethod
    def validate(self, graph: Any, context: Dict[str, Any]) -> ValidationResult:
        """
        Validate the graph.

        Args:
            graph: The execution graph to validate
            context: Additional context (memory limits, device info, etc.)

        Returns:
            ValidationResult with pass/fail and diagnostics

        Note:
            Implementations should NOT raise exceptions for validation failures.
            Instead, return ValidationResult(passed=False, ...).
            Exceptions should only be raised for internal errors.
        """
        pass

    def should_run(self, safety_level: SafetyLevel) -> bool:
        """
        Check if this validator should run at the given safety level.

        Override in subclasses to customize when validator runs.
        Default: run at all levels.
        """
        return True
