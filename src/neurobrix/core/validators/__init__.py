"""NeuroBrix validators: the container's structural check (`nbx_validator`, read by `neurobrix validate`) and the
compiled branch's tensor checks (`tensor_validator`).

A layered validation pipeline (fusion, memory, numerical, op, shape and sync validators, runtime guards) stood here
until 2026-10-30, re-exported by this file and called by nothing in the engine, its tools or its tests: deleted.
"""

from .tensor_validator import TensorValidator
from .nbx_validator import NBXValidator, validate_nbx, ValidationLevel, NBXValidationError

__all__ = ["TensorValidator", "NBXValidator", "validate_nbx", "ValidationLevel", "NBXValidationError"]
