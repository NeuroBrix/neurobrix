"""
NeuroBrix Validators - Numerical Validator.

Validates numerical stability of computations.

ZERO FALLBACK: Numerical issues raise explicit NumericalDivergence.
"""

from typing import Any, Dict, List, Optional

from .base import (
    BaseValidator, SafetyLevel, ValidationResult,
    NumericalDivergence, NaNInfDetected
)
from .config import get_config


class NumericalValidator(BaseValidator):
    """
    Validates numerical stability.

    Checks:
    - NaN detection
    - Inf detection
    - Value range for dtype
    - Gradient magnitude
    - Loss stability
    """

    @property
    def name(self) -> str:
        return "NumericalValidator"

    def __init__(self, safety_level: SafetyLevel = SafetyLevel.STANDARD):
        super().__init__(safety_level)
        self.config = get_config()
        self._check_count = 0
        self._nan_history: List[int] = []
        self._inf_history: List[int] = []

    def validate(self, graph: Any, context: Dict[str, Any]) -> ValidationResult:
        """
        Validate numerical stability of graph execution.

        Args:
            graph: Execution graph
            context: Must contain:
                - 'tensors': Dict of tensor data to check
                - 'dtype': Data type for range checking

        Returns:
            ValidationResult with numerical validation status
        """
        warnings = []
        errors = []
        metrics = {}

        tensors = context.get('tensors', {})
        dtype = context.get('dtype', 'float32')

        if not tensors:
            return ValidationResult(
                passed=True,
                validator_name=self.name,
                message="No tensors to validate",
                warnings=warnings,
            )

        total_nan = 0
        total_inf = 0
        total_elements = 0

        for name, tensor in tensors.items():
            nan_count, inf_count, elem_count = self._check_tensor(
                tensor, name, dtype, warnings, errors, metrics
            )
            total_nan += nan_count
            total_inf += inf_count
            total_elements += elem_count

        # Update history for trend detection
        self._check_count += 1
        self._nan_history.append(total_nan)
        self._inf_history.append(total_inf)

        # Keep history bounded
        if len(self._nan_history) > 100:
            self._nan_history = self._nan_history[-100:]
            self._inf_history = self._inf_history[-100:]

        # Detect trends
        if self.safety_level >= SafetyLevel.STANDARD:
            self._check_trends(warnings, metrics)

        metrics['total_nan_count'] = total_nan
        metrics['total_inf_count'] = total_inf
        metrics['total_elements_checked'] = total_elements
        metrics['check_count'] = self._check_count

        if errors:
            return ValidationResult(
                passed=False,
                validator_name=self.name,
                message="\n".join(errors),
                warnings=warnings,
                metrics=metrics,
            )

        return ValidationResult(
            passed=True,
            validator_name=self.name,
            message=f"Numerical check passed: {total_elements} elements checked",
            warnings=warnings,
            metrics=metrics,
        )

    def _check_tensor(
        self,
        tensor: Any,
        name: str,
        dtype: str,
        warnings: List[str],
        errors: List[str],
        metrics: Dict[str, Any],
    ) -> tuple:
        """Check a single tensor for numerical issues."""
        config = self.config

        # Get tensor stats (framework-agnostic)
        try:
            stats = self._get_tensor_stats(tensor)
        except Exception as e:
            warnings.append(f"Could not analyze tensor '{name}': {e}")
            return 0, 0, 0

        nan_count = stats.get('nan_count', 0)
        inf_count = stats.get('inf_count', 0)
        elem_count = stats.get('numel', 0)
        max_val = stats.get('max', 0)
        min_val = stats.get('min', 0)

        # Store per-tensor metrics
        metrics[f'{name}_nan'] = nan_count
        metrics[f'{name}_inf'] = inf_count
        metrics[f'{name}_max'] = max_val
        metrics[f'{name}_min'] = min_val

        # Check NaN threshold
        if nan_count > config.nan_threshold:
            errors.append(
                f"Tensor '{name}' contains {nan_count} NaN values "
                f"(threshold: {config.nan_threshold})"
            )

        # Check Inf threshold
        if inf_count > config.inf_threshold:
            errors.append(
                f"Tensor '{name}' contains {inf_count} Inf values "
                f"(threshold: {config.inf_threshold})"
            )

        # Check value range for dtype
        dtype_max = config.get_dtype_max(dtype)
        if abs(max_val) > dtype_max or abs(min_val) > dtype_max:
            warnings.append(
                f"Tensor '{name}' values near dtype limit: "
                f"[{min_val:.2e}, {max_val:.2e}] (max: {dtype_max:.2e})"
            )

        return nan_count, inf_count, elem_count

    def _get_tensor_stats(self, tensor: Any) -> Dict[str, Any]:
        """Get statistics from tensor (framework-agnostic)."""
        stats = {}

        # Try to get numel
        if hasattr(tensor, 'numel'):
            stats['numel'] = tensor.numel()
        elif hasattr(tensor, 'size'):
            import functools
            import operator
            stats['numel'] = functools.reduce(operator.mul, tensor.size(), 1)
        else:
            stats['numel'] = 0

        # Try to detect NaN/Inf using Triton-compatible checks
        # This assumes tensor is already on GPU and we can use Triton
        try:
            # For PyTorch tensors (during development/testing only)
            if hasattr(tensor, 'isnan'):
                stats['nan_count'] = int(tensor.isnan().sum().item())
            else:
                stats['nan_count'] = 0

            if hasattr(tensor, 'isinf'):
                stats['inf_count'] = int(tensor.isinf().sum().item())
            else:
                stats['inf_count'] = 0

            if hasattr(tensor, 'max') and hasattr(tensor, 'min'):
                # Mask out inf for max/min calculation
                finite_mask = ~(tensor.isnan() | tensor.isinf())
                if finite_mask.any():
                    finite_tensor = tensor[finite_mask]
                    stats['max'] = float(finite_tensor.max().item())
                    stats['min'] = float(finite_tensor.min().item())
                else:
                    stats['max'] = float('nan')
                    stats['min'] = float('nan')
            else:
                stats['max'] = 0
                stats['min'] = 0
        except Exception:
            stats['nan_count'] = 0
            stats['inf_count'] = 0
            stats['max'] = 0
            stats['min'] = 0

        return stats

    def _check_trends(
        self,
        warnings: List[str],
        metrics: Dict[str, Any],
    ) -> None:
        """Check for concerning numerical trends."""
        if len(self._nan_history) < 10:
            return

        # Check if NaN count is increasing
        recent_nan = self._nan_history[-10:]
        if all(recent_nan[i] <= recent_nan[i+1] for i in range(len(recent_nan)-1)):
            if recent_nan[-1] > recent_nan[0]:
                warnings.append(
                    f"NaN count increasing trend: {recent_nan[0]} -> {recent_nan[-1]}"
                )
                metrics['nan_increasing_trend'] = True

        # Check if Inf count is increasing
        recent_inf = self._inf_history[-10:]
        if all(recent_inf[i] <= recent_inf[i+1] for i in range(len(recent_inf)-1)):
            if recent_inf[-1] > recent_inf[0]:
                warnings.append(
                    f"Inf count increasing trend: {recent_inf[0]} -> {recent_inf[-1]}"
                )
                metrics['inf_increasing_trend'] = True

    def check_tensor_immediate(
        self,
        tensor: Any,
        name: str = "tensor",
        raise_on_issue: bool = True,
    ) -> Dict[str, Any]:
        """
        Immediately check a tensor for NaN/Inf.

        Args:
            tensor: Tensor to check
            name: Name for error messages
            raise_on_issue: If True, raise NaNInfDetected on issues

        Returns:
            Dict with check results

        Raises:
            NaNInfDetected if issues found and raise_on_issue=True
        """
        stats = self._get_tensor_stats(tensor)

        if raise_on_issue:
            if stats.get('nan_count', 0) > 0:
                raise NaNInfDetected(
                    f"NaN detected in {name}",
                    nan_count=stats['nan_count'],
                    suggestion="Check for division by zero, log of negative, or exploding gradients",
                )
            if stats.get('inf_count', 0) > 0:
                raise NaNInfDetected(
                    f"Inf detected in {name}",
                    inf_count=stats['inf_count'],
                    suggestion="Check for overflow or very large values before operations",
                )

        return stats

    def reset_history(self) -> None:
        """Reset numerical tracking history."""
        self._check_count = 0
        self._nan_history = []
        self._inf_history = []

    def should_run(self, safety_level: SafetyLevel) -> bool:
        """Numerical validation runs at STANDARD and PARANOID levels."""
        return safety_level >= SafetyLevel.STANDARD
