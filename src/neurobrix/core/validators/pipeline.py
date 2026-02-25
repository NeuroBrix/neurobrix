"""
NeuroBrix Validators - Validation Pipeline.

Orchestrates all validators in correct order.

ZERO FALLBACK: Pipeline aggregates all validation results.
"""

from typing import Any, Dict, List, Optional, Type

from .base import BaseValidator, SafetyLevel, ValidationResult, ValidationError
from .config import get_config, ValidatorConfig
from .shape_validator import ShapeValidator
from .memory_validator import MemoryValidator
from .numerical_validator import NumericalValidator
from .op_validator import OpValidator
from .sync_validator import SyncValidator
from .fusion_validator import FusionValidator


class ValidationPipeline:
    """
    Orchestrates validation across all validators.

    Runs validators in correct order:
    1. OpValidator - Check all ops are supported
    2. ShapeValidator - Check shape compatibility
    3. MemoryValidator - Check memory feasibility
    4. SyncValidator - Check synchronization
    5. FusionValidator - Check fusion correctness
    6. NumericalValidator - Check numerical stability (runtime)

    Can run in different modes:
    - Full validation (all validators)
    - Import-time validation (ops, shapes)
    - Compile-time validation (memory, sync, fusion)
    - Runtime validation (numerical)
    """

    # Validation phases
    IMPORT_TIME_VALIDATORS = [OpValidator, ShapeValidator]
    COMPILE_TIME_VALIDATORS = [MemoryValidator, SyncValidator, FusionValidator]
    RUNTIME_VALIDATORS = [NumericalValidator]

    def __init__(
        self,
        level: SafetyLevel = SafetyLevel.STANDARD,
        config: Optional[ValidatorConfig] = None,
    ):
        self.level = level
        self.config = config or get_config()

        # Initialize all validators
        self.validators: Dict[str, BaseValidator] = {
            'op': OpValidator(level),
            'shape': ShapeValidator(level),
            'memory': MemoryValidator(level),
            'sync': SyncValidator(level),
            'fusion': FusionValidator(level),
            'numerical': NumericalValidator(level),
        }

        # Results cache
        self._results: List[ValidationResult] = []

    def validate_all(
        self,
        graph: Any,
        context: Dict[str, Any],
    ) -> List[ValidationResult]:
        """
        Run all validators on the graph.

        Args:
            graph: Execution graph
            context: Validation context with:
                - input_shapes: Dict of tensor shapes
                - memory_plan: Memory allocation plan
                - device_assignments: Device mapping
                - fusion_groups: Fusion plan
                - available_memory: Available VRAM
                - etc.

        Returns:
            List of ValidationResults from all validators
        """
        results = []

        for name, validator in self.validators.items():
            if validator.should_run(self.level):
                try:
                    result = validator.validate(graph, context)
                    results.append(result)
                except Exception as e:
                    results.append(ValidationResult(
                        passed=False,
                        validator_name=validator.name,
                        message=f"Validator error: {e}",
                    ))

        self._results = results
        return results

    def validate_topology(
        self,
        graph: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationResult]:
        """
        Run import-time validators (ops, shapes).

        Called when graph is first created/imported.
        """
        context = context or {}
        results = []

        for validator_cls in self.IMPORT_TIME_VALIDATORS:
            validator = self._get_validator_by_class(validator_cls)
            if validator and validator.should_run(self.level):
                result = validator.validate(graph, context)
                results.append(result)

        self._results = results
        return results

    def validate_hardware_plan(
        self,
        plan: Any,
        hardware: Dict[str, Any],
    ) -> List[ValidationResult]:
        """
        Run compile-time validators (memory, sync, fusion).

        Called when hardware plan is generated.
        """
        context = {
            'memory_plan': getattr(plan, 'memory_plan', {}),
            'device_assignments': getattr(plan, 'device_assignments', {}),
            'stream_assignments': getattr(plan, 'stream_assignments', {}),
            'fusion_groups': getattr(plan, 'fusion_groups', []),
            'available_memory': hardware.get('available_memory', 0),
            'device_info': hardware,
        }

        results = []
        for validator_cls in self.COMPILE_TIME_VALIDATORS:
            validator = self._get_validator_by_class(validator_cls)
            if validator and validator.should_run(self.level):
                result = validator.validate(plan, context)
                results.append(result)

        self._results = results
        return results

    def validate_execution_plan(
        self,
        exec_plan: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationResult]:
        """
        Validate complete execution plan.

        Combines topology and hardware validation.
        """
        context = context or {}

        # Extract context from execution plan
        if hasattr(exec_plan, 'graph'):
            context['graph'] = exec_plan.graph
        if hasattr(exec_plan, 'memory_plan'):
            context['memory_plan'] = exec_plan.memory_plan
        if hasattr(exec_plan, 'device_map'):
            context['device_assignments'] = exec_plan.device_map

        return self.validate_all(exec_plan, context)

    def validate_runtime(
        self,
        tensors: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationResult]:
        """
        Run runtime validators (numerical stability).

        Called during execution to check intermediate values.
        """
        context = context or {}
        context['tensors'] = tensors

        results = []
        for validator_cls in self.RUNTIME_VALIDATORS:
            validator = self._get_validator_by_class(validator_cls)
            if validator and validator.should_run(self.level):
                result = validator.validate(None, context)
                results.append(result)

        self._results = results
        return results

    def _get_validator_by_class(
        self,
        validator_cls: Type[BaseValidator],
    ) -> Optional[BaseValidator]:
        """Get validator instance by class type."""
        for validator in self.validators.values():
            if isinstance(validator, validator_cls):
                return validator
        return None

    # ========================================================================
    # Result Aggregation
    # ========================================================================

    def all_passed(self) -> bool:
        """Check if all validations passed."""
        return all(r.passed for r in self._results)

    def get_errors(self) -> List[str]:
        """Get all error messages."""
        errors = []
        for result in self._results:
            if not result.passed:
                errors.append(f"[{result.validator_name}] {result.message}")
        return errors

    def get_warnings(self) -> List[str]:
        """Get all warnings."""
        warnings = []
        for result in self._results:
            warnings.extend(
                f"[{result.validator_name}] {w}" for w in result.warnings
            )
        return warnings

    def raise_if_failed(self) -> None:
        """Raise ValidationError if any validation failed."""
        if not self.all_passed():
            errors = self.get_errors()
            raise ValidationError(
                f"Validation failed:\n" + "\n".join(errors)
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            'all_passed': self.all_passed(),
            'validator_count': len(self._results),
            'passed_count': sum(1 for r in self._results if r.passed),
            'failed_count': sum(1 for r in self._results if not r.passed),
            'warning_count': sum(len(r.warnings) for r in self._results),
            'safety_level': self.level.name,
            'errors': self.get_errors(),
            'warnings': self.get_warnings(),
        }

    # ========================================================================
    # Configuration
    # ========================================================================

    def set_level(self, level: SafetyLevel) -> None:
        """Update safety level for all validators."""
        self.level = level
        for validator in self.validators.values():
            validator.safety_level = level

    def add_validator(
        self,
        name: str,
        validator: BaseValidator,
    ) -> None:
        """Add custom validator to pipeline."""
        self.validators[name] = validator

    def remove_validator(self, name: str) -> None:
        """Remove validator from pipeline."""
        if name in self.validators:
            del self.validators[name]

    def get_validator(self, name: str) -> Optional[BaseValidator]:
        """Get validator by name."""
        return self.validators.get(name)
