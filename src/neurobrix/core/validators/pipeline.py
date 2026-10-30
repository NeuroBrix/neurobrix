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


    # ========================================================================
    # Configuration
    # ========================================================================


