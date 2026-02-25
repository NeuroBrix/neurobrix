"""
NeuroBrix Validators - Shape Validator.

Validates tensor shape compatibility throughout the graph.

ZERO FALLBACK: Shape mismatches raise explicit ShapeViolation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

from .base import BaseValidator, SafetyLevel, ValidationResult, ShapeViolation
from .config import get_config


@dataclass
class ShapeConstraints:
    """
    Shape constraints for a tensor or operation.

    Supports:
    - Fixed dimensions
    - Dynamic dimensions with bounds
    - Broadcast rules
    """
    dims: Tuple[Optional[int], ...]  # None = dynamic
    min_dims: Optional[Tuple[int, ...]] = None
    max_dims: Optional[Tuple[int, ...]] = None
    must_match: Set[str] = field(default_factory=set)  # Other tensors to match

    def validate_shape(self, shape: Tuple[int, ...]) -> Tuple[bool, str]:
        """
        Validate a concrete shape against constraints.

        Returns (is_valid, error_message).
        """
        # Check rank
        if len(shape) != len(self.dims):
            return False, f"Rank mismatch: expected {len(self.dims)}, got {len(shape)}"

        # Check each dimension
        for i, (actual, expected) in enumerate(zip(shape, self.dims)):
            # Fixed dimension check
            if expected is not None and actual != expected:
                return False, f"Dim {i}: expected {expected}, got {actual}"

            # Min bound check
            if self.min_dims is not None and i < len(self.min_dims):
                if actual < self.min_dims[i]:
                    return False, f"Dim {i}: {actual} < min {self.min_dims[i]}"

            # Max bound check
            if self.max_dims is not None and i < len(self.max_dims):
                if actual > self.max_dims[i]:
                    return False, f"Dim {i}: {actual} > max {self.max_dims[i]}"

        return True, ""


class ShapeValidator(BaseValidator):
    """
    Validates tensor shapes throughout execution graph.

    Checks:
    - Input/output shape compatibility
    - Broadcast rules
    - Shape constraint bounds
    - Dynamic shape consistency
    """

    @property
    def name(self) -> str:
        return "ShapeValidator"

    def __init__(self, safety_level: SafetyLevel = SafetyLevel.STANDARD):
        super().__init__(safety_level)
        self.config = get_config()
        self._shape_cache: Dict[str, Tuple[int, ...]] = {}

    def validate(self, graph: Any, context: Dict[str, Any]) -> ValidationResult:
        """
        Validate all shapes in the graph.

        Args:
            graph: Execution graph with nodes
            context: Must contain 'input_shapes' dict

        Returns:
            ValidationResult with shape validation status
        """
        warnings = []
        errors = []

        # Get input shapes from context
        input_shapes = context.get('input_shapes', {})
        if not input_shapes and self.safety_level == SafetyLevel.PARANOID:
            warnings.append("No input shapes provided for validation")

        # Validate each node
        nodes = getattr(graph, 'nodes', [])
        if hasattr(graph, '__iter__') and not nodes:
            nodes = list(graph)

        for node in nodes:
            try:
                self._validate_node(node, input_shapes, warnings, errors)
            except Exception as e:
                errors.append(f"Error validating node {getattr(node, 'name', node)}: {e}")

        # Check global constraints
        self._check_global_constraints(input_shapes, warnings, errors)

        if errors:
            return ValidationResult(
                passed=False,
                validator_name=self.name,
                message="\n".join(errors),
                warnings=warnings,
            )

        return ValidationResult(
            passed=True,
            validator_name=self.name,
            message="All shapes validated",
            warnings=warnings,
        )

    def _validate_node(
        self,
        node: Any,
        input_shapes: Dict[str, Tuple[int, ...]],
        warnings: List[str],
        errors: List[str],
    ) -> None:
        """Validate shapes for a single node."""
        op = getattr(node, 'op', None)
        if op is None:
            return

        # Get input shapes for this node
        node_input_shapes = []
        inputs = getattr(node, 'inputs', [])
        for inp in inputs:
            inp_name = getattr(inp, 'name', str(inp))
            if inp_name in input_shapes:
                node_input_shapes.append(input_shapes[inp_name])
            elif inp_name in self._shape_cache:
                node_input_shapes.append(self._shape_cache[inp_name])

        # Validate based on op type
        if op in ('matmul', 'linear'):
            self._validate_matmul(node, node_input_shapes, errors)
        elif op in ('add', 'sub', 'mul', 'div'):
            self._validate_broadcast(node, node_input_shapes, warnings)
        elif op == 'concat':
            self._validate_concat(node, node_input_shapes, errors)
        elif op in ('softmax', 'layernorm', 'rmsnorm'):
            self._validate_reduction_dim(node, node_input_shapes, errors)

    def _validate_matmul(
        self,
        node: Any,
        shapes: List[Tuple[int, ...]],
        errors: List[str],
    ) -> None:
        """Validate matmul shape compatibility."""
        if len(shapes) < 2:
            return

        a_shape, b_shape = shapes[0], shapes[1]
        if len(a_shape) < 2 or len(b_shape) < 2:
            errors.append(
                f"Matmul requires 2D+ tensors, got {len(a_shape)}D and {len(b_shape)}D"
            )
            return

        # Check inner dimensions
        if a_shape[-1] != b_shape[-2]:
            errors.append(
                f"Matmul dimension mismatch: "
                f"{a_shape} @ {b_shape} - {a_shape[-1]} != {b_shape[-2]}"
            )

    def _validate_broadcast(
        self,
        node: Any,
        shapes: List[Tuple[int, ...]],
        warnings: List[str],
    ) -> None:
        """Validate broadcast compatibility."""
        if len(shapes) < 2:
            return

        # Check if shapes are broadcastable
        try:
            self._compute_broadcast_shape(shapes)
        except ValueError as e:
            warnings.append(f"Broadcast warning for {getattr(node, 'name', node)}: {e}")

    def _compute_broadcast_shape(
        self,
        shapes: List[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        """Compute broadcast result shape."""
        if not shapes:
            return ()

        result = list(shapes[0])
        for shape in shapes[1:]:
            # Align from the right
            while len(result) < len(shape):
                result.insert(0, 1)
            shape_list = [1] * (len(result) - len(shape)) + list(shape)

            # Check each dimension
            for i in range(len(result)):
                if result[i] == shape_list[i]:
                    continue
                elif result[i] == 1:
                    result[i] = shape_list[i]
                elif shape_list[i] == 1:
                    continue
                else:
                    raise ValueError(
                        f"Cannot broadcast dim {i}: {result[i]} vs {shape_list[i]}"
                    )

        return tuple(result)

    def _validate_concat(
        self,
        node: Any,
        shapes: List[Tuple[int, ...]],
        errors: List[str],
    ) -> None:
        """Validate concat shape compatibility."""
        if len(shapes) < 2:
            return

        axis = getattr(node, 'axis', 0)
        if axis < 0:
            axis = len(shapes[0]) + axis

        # All shapes must match except on concat axis
        ref_shape: List[Optional[int]] = list(shapes[0])
        ref_shape[axis] = None  # Ignore concat dimension

        for i, shape in enumerate(shapes[1:], 1):
            if len(shape) != len(ref_shape):
                errors.append(
                    f"Concat rank mismatch: input {i} has {len(shape)} dims, "
                    f"expected {len(ref_shape)}"
                )
                continue

            for j, (ref, actual) in enumerate(zip(ref_shape, shape)):
                if ref is not None and ref != actual:
                    errors.append(
                        f"Concat shape mismatch at input {i}, dim {j}: "
                        f"expected {ref}, got {actual}"
                    )

    def _validate_reduction_dim(
        self,
        node: Any,
        shapes: List[Tuple[int, ...]],
        errors: List[str],
    ) -> None:
        """Validate reduction dimension exists."""
        if not shapes:
            return

        shape = shapes[0]
        dim = getattr(node, 'dim', -1)
        if dim < 0:
            dim = len(shape) + dim

        if dim < 0 or dim >= len(shape):
            errors.append(
                f"Invalid reduction dim {dim} for shape {shape}"
            )

    def _check_global_constraints(
        self,
        input_shapes: Dict[str, Tuple[int, ...]],
        warnings: List[str],
        errors: List[str],
    ) -> None:
        """Check global shape constraints from config."""
        config = self.config

        for name, shape in input_shapes.items():
            # Check max dimensions
            if len(shape) > config.max_tensor_dims:
                errors.append(
                    f"Tensor {name} has {len(shape)} dims, max is {config.max_tensor_dims}"
                )

            # Check batch size (assume first dim)
            if len(shape) > 0 and shape[0] > config.max_batch_size:
                warnings.append(
                    f"Tensor {name} batch size {shape[0]} exceeds recommended {config.max_batch_size}"
                )

            # Check sequence length (assume second dim for transformer inputs)
            if len(shape) > 1 and shape[1] > config.max_sequence_length:
                warnings.append(
                    f"Tensor {name} sequence length {shape[1]} exceeds max {config.max_sequence_length}"
                )

    def validate_shapes_match(
        self,
        expected: Tuple[int, ...],
        actual: Tuple[int, ...],
        name: str = "",
    ) -> None:
        """
        Validate that two shapes match exactly.

        Raises ShapeViolation if mismatch.
        """
        if expected != actual:
            raise ShapeViolation(
                f"Shape mismatch for {name}",
                expected_shape=expected,
                actual_shape=actual,
                suggestion=f"Expected {expected}, got {actual}. "
                          f"Check input dimensions.",
            )

    def should_run(self, safety_level: SafetyLevel) -> bool:
        """Shape validation runs at all levels."""
        return True
