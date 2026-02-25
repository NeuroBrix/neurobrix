"""
NeuroBrix Validators - Defensive Programming Layer.

PRINCIPE: "Fail LOUD, Fail EARLY, Fail with ACTIONABLE message"

Validators are organized in layers:
1. IMPORT TIME: Op coverage, shape bounds
2. COMPILE TIME: Memory plan, dependencies, fusion correctness
3. RUNTIME: Input shapes, NaN/Inf detection, memory bounds

Usage:
    from neurobrix.core.validators import ValidationPipeline, SafetyLevel

    pipeline = ValidationPipeline(level=SafetyLevel.STANDARD)
    pipeline.validate_topology(topology)
    pipeline.validate_hardware_plan(plan, hardware)
    pipeline.validate_execution_plan(exec_plan)

ZERO FALLBACK: All validation failures raise explicit exceptions.
"""

from .base import (
    # Enums
    SafetyLevel,
    # Base classes
    BaseValidator,
    ValidationResult,
    # Exceptions
    ValidationError,
    ShapeViolation,
    MemoryPlanError,
    NumericalDivergence,
    UnsupportedOpsError,
    SyncError,
    FusionError,
    NaNInfDetected,
    DeadlockRisk,
)

from .config import (
    ValidatorConfig,
    get_config,
    set_config,
    reset_config,
    update_config,
)

from .shape_validator import ShapeValidator, ShapeConstraints
from .memory_validator import MemoryValidator
from .numerical_validator import NumericalValidator
from .op_validator import OpValidator
from .sync_validator import SyncValidator
from .fusion_validator import FusionValidator
from .runtime_guards import RuntimeGuards
from .pipeline import ValidationPipeline
from .tensor_validator import TensorValidator
from .nbx_validator import (
    NBXValidator,
    validate_nbx,
    ValidationLevel,
    NBXValidationError,
    ValidationResult as NBXValidationResult,
)

__all__ = [
    # Enums
    "SafetyLevel",

    # Base classes
    "BaseValidator",
    "ValidationResult",

    # Exceptions
    "ValidationError",
    "ShapeViolation",
    "MemoryPlanError",
    "NumericalDivergence",
    "UnsupportedOpsError",
    "SyncError",
    "FusionError",
    "NaNInfDetected",
    "DeadlockRisk",

    # Config
    "ValidatorConfig",
    "get_config",
    "set_config",
    "reset_config",
    "update_config",

    # Validators
    "ShapeValidator",
    "ShapeConstraints",
    "MemoryValidator",
    "NumericalValidator",
    "OpValidator",
    "SyncValidator",
    "FusionValidator",
    "RuntimeGuards",
    "TensorValidator",

    # Pipeline
    "ValidationPipeline",

    # NBX Validator
    "NBXValidator",
    "validate_nbx",
    "ValidationLevel",
    "NBXValidationError",
    "NBXValidationResult",
]
