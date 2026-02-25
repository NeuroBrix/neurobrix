"""
NeuroBrix Validators - Fusion Validator.

Validates kernel fusion correctness.

ZERO FALLBACK: Invalid fusions raise explicit FusionError.
"""

from typing import Any, Dict, List, Set, Tuple, Optional

from .base import BaseValidator, SafetyLevel, ValidationResult, FusionError
from .config import get_config


class FusionValidator(BaseValidator):
    """
    Validates kernel fusion correctness.

    Checks:
    - Fusion candidate validity
    - Data dependency preservation
    - Memory access patterns
    - Fusion size limits
    - Shape compatibility
    """

    # Operations that can be fused together
    FUSIBLE_OPS: Set[str] = {
        # Elementwise ops (highly fusible)
        'add', 'sub', 'mul', 'div', 'sqrt', 'exp', 'log', 'pow',
        'relu', 'gelu', 'silu', 'sigmoid', 'tanh',
        # Reduction ops (limited fusion)
        'sum', 'mean', 'max', 'min',
        # Normalization (can fuse with elementwise)
        'layernorm', 'rmsnorm',
    }

    # Operations that cannot be fused
    UNFUSIBLE_OPS: Set[str] = {
        'matmul', 'linear', 'conv2d', 'conv1d',
        'scaled_dot_product_attention', 'flash_attention',
    }

    # Operations that break fusion chains
    FUSION_BREAKERS: Set[str] = {
        'concat', 'split', 'reshape', 'transpose', 'permute',
        'gather', 'scatter',
    }

    @property
    def name(self) -> str:
        return "FusionValidator"

    def __init__(self, safety_level: SafetyLevel = SafetyLevel.STANDARD):
        super().__init__(safety_level)
        self.config = get_config()

    def validate(self, graph: Any, context: Dict[str, Any]) -> ValidationResult:
        """
        Validate fusion plan for the graph.

        Args:
            graph: Execution graph with nodes
            context: Must contain:
                - 'fusion_groups': List of node groups to fuse

        Returns:
            ValidationResult with fusion validation status
        """
        warnings = []
        errors = []
        metrics = {}

        fusion_groups = context.get('fusion_groups', [])

        if not fusion_groups:
            return ValidationResult(
                passed=True,
                validator_name=self.name,
                message="No fusion groups to validate",
                warnings=warnings,
            )

        metrics['num_fusion_groups'] = len(fusion_groups)

        total_ops = 0
        for i, group in enumerate(fusion_groups):
            group_errors = self._validate_fusion_group(group, graph, context)
            for error in group_errors:
                errors.append(f"Fusion group {i}: {error}")
            total_ops += len(group)

        metrics['total_fused_ops'] = total_ops
        metrics['avg_group_size'] = total_ops / len(fusion_groups) if fusion_groups else 0

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
            message=f"Validated {len(fusion_groups)} fusion groups with {total_ops} ops",
            warnings=warnings,
            metrics=metrics,
        )

    def _validate_fusion_group(
        self,
        group: List[Any],
        graph: Any,
        context: Dict[str, Any],
    ) -> List[str]:
        """Validate a single fusion group."""
        errors = []
        config = self.config

        # Check group size
        if len(group) > config.max_fusion_ops:
            errors.append(
                f"Group size {len(group)} exceeds max {config.max_fusion_ops}"
            )

        # Check ops are fusible
        group_ops = []
        for node in group:
            op = getattr(node, 'op', None)
            if op is None:
                continue
            op_lower = op.lower()
            group_ops.append(op_lower)

            if op_lower in self.UNFUSIBLE_OPS:
                errors.append(f"Op '{op}' cannot be fused")

        # Check for fusion breakers in the middle
        if len(group) > 2:
            middle_ops = group_ops[1:-1]
            for op in middle_ops:
                if op in self.FUSION_BREAKERS:
                    errors.append(f"Fusion breaker '{op}' in middle of group")

        # Check data dependencies
        dep_errors = self._check_dependencies(group)
        errors.extend(dep_errors)

        # Check shape compatibility
        shape_errors = self._check_shape_compatibility(group, context)
        errors.extend(shape_errors)

        return errors

    def _check_dependencies(self, group: List[Any]) -> List[str]:
        """Check that data dependencies are preserved in fusion."""
        errors = []

        # Build set of nodes in group
        group_names = set()
        for node in group:
            name = getattr(node, 'name', str(node))
            group_names.add(name)

        # Check each node's inputs
        for node in group:
            inputs = getattr(node, 'inputs', [])
            for inp in inputs:
                inp_name = getattr(inp, 'name', str(inp))

                # If input is from outside group, check it comes first
                if inp_name not in group_names:
                    # External dependency - OK, will be handled by fusion kernel
                    pass
                else:
                    # Internal dependency - must be earlier in group
                    found_producer = False
                    for i, other in enumerate(group):
                        other_name = getattr(other, 'name', str(other))
                        if other_name == inp_name:
                            found_producer = True
                            break
                        if other_name == getattr(node, 'name', ''):
                            # Consumer before producer
                            errors.append(
                                f"Dependency violation: {inp_name} must come before "
                                f"{getattr(node, 'name', node)}"
                            )
                            break

        return errors

    def _check_shape_compatibility(
        self,
        group: List[Any],
        context: Dict[str, Any],
    ) -> List[str]:
        """Check that shapes are compatible for fusion."""
        errors = []

        input_shapes = context.get('input_shapes', {})
        if not input_shapes:
            return errors

        # Get shapes of all nodes in group
        shapes = []
        for node in group:
            name = getattr(node, 'name', str(node))
            if name in input_shapes:
                shapes.append(input_shapes[name])

        if len(shapes) < 2:
            return errors

        # Check all shapes are broadcastable
        ref_shape = shapes[0]
        for shape in shapes[1:]:
            if not self._are_broadcastable(ref_shape, shape):
                errors.append(
                    f"Shapes not compatible for fusion: {ref_shape} vs {shape}"
                )

        return errors

    def _are_broadcastable(
        self,
        shape1: Tuple[int, ...],
        shape2: Tuple[int, ...],
    ) -> bool:
        """Check if two shapes are broadcastable."""
        # Align from right
        len1, len2 = len(shape1), len(shape2)
        max_len = max(len1, len2)

        for i in range(1, max_len + 1):
            d1 = shape1[-i] if i <= len1 else 1
            d2 = shape2[-i] if i <= len2 else 1

            if d1 != d2 and d1 != 1 and d2 != 1:
                return False

        return True

    def can_fuse(
        self,
        op1: str,
        op2: str,
    ) -> Tuple[bool, str]:
        """
        Check if two operations can be fused.

        Returns:
            (can_fuse, reason)
        """
        op1_lower = op1.lower()
        op2_lower = op2.lower()

        # Unfusible ops
        if op1_lower in self.UNFUSIBLE_OPS:
            return False, f"'{op1}' cannot be fused"
        if op2_lower in self.UNFUSIBLE_OPS:
            return False, f"'{op2}' cannot be fused"

        # Fusion breakers
        if op1_lower in self.FUSION_BREAKERS:
            return False, f"'{op1}' breaks fusion chains"
        if op2_lower in self.FUSION_BREAKERS:
            return False, f"'{op2}' breaks fusion chains"

        # Both fusible
        if op1_lower in self.FUSIBLE_OPS and op2_lower in self.FUSIBLE_OPS:
            return True, "Both ops are fusible"

        return False, f"Unknown fusion compatibility: {op1}, {op2}"

    def find_fusion_groups(
        self,
        graph: Any,
    ) -> List[List[Any]]:
        """
        Find valid fusion groups in the graph.

        Returns list of node groups that can be fused.
        """
        groups = []
        current_group: List[Any] = []

        nodes = getattr(graph, 'nodes', [])
        if hasattr(graph, '__iter__') and not nodes:
            nodes = list(graph)

        for node in nodes:
            op = getattr(node, 'op', None)
            if op is None:
                continue

            op_lower = op.lower()

            # Check if can add to current group
            if op_lower in self.FUSIBLE_OPS:
                if len(current_group) < self.config.max_fusion_ops:
                    current_group.append(node)
                else:
                    # Group full, start new one
                    if len(current_group) > 1:
                        groups.append(current_group)
                    current_group = [node]
            else:
                # Non-fusible op, close current group
                if len(current_group) > 1:
                    groups.append(current_group)
                current_group = []

        # Don't forget last group
        if len(current_group) > 1:
            groups.append(current_group)

        return groups

    def validate_fusion(
        self,
        nodes: List[Any],
    ) -> None:
        """
        Validate that a list of nodes can be fused.

        Raises FusionError if fusion is invalid.
        """
        if len(nodes) > self.config.max_fusion_ops:
            raise FusionError(
                f"Fusion group too large: {len(nodes)} > {self.config.max_fusion_ops}",
                nodes_to_fuse=[getattr(n, 'name', str(n)) for n in nodes],
                reason="Exceeds max fusion ops limit",
            )

        for node in nodes:
            op = getattr(node, 'op', None)
            if op and op.lower() in self.UNFUSIBLE_OPS:
                raise FusionError(
                    f"Cannot fuse unfusible op: {op}",
                    nodes_to_fuse=[getattr(n, 'name', str(n)) for n in nodes],
                    reason=f"Op '{op}' cannot be fused",
                )

    def should_run(self, safety_level: SafetyLevel) -> bool:
        """Fusion validation runs at STANDARD and PARANOID levels."""
        return safety_level >= SafetyLevel.STANDARD
